clear all; close all; clc;
%% Initialization

%Load all the important files for the standard (if it doesn't work please insert your directory or add your path!)
Gamma = load('.\Gamma.txt');% Compound Quantization function
RA_shift12 = dlmread('.\RA_shift12.txt');   % RA_shift12 table
RA_shift = dlmread('.\RA_shift.txt');   % RA_shift table

pcm_input='.\speech.wav'; %Choose the directory of the audio file that you want to compress




%Read the pcm input file
[audio_signal,Fs] = audioread(pcm_input);
track_info = audioinfo(pcm_input);
[Rows, Columns]=size(audio_signal);

if Columns==1 
    fprintf( '\nThe audio signal is mono.');
    audio_normalized=audio_signal*(2^15); %the input audio_signal is 16-bit, little-endian PCM represented in fractional so it needs to be shifted left 16 times
elseif Columns==2
    fprintf( '\nThe audio signal is stereo.');
    [Mid, Side]=Channel_Decorrelation(audio_signal);
   	audio_normalized=Mid*(2^16); %the input audio_signal is 16-bit, little-endian PCM represented in fractional so it needs to be shifted left 16 times
else
    fprintf('No code yet for more than 2 channels')
end


prompt = '\n\n\nPress: \n"1" for Arithmetic Coding with the actual pmf of the source.\n"2" for Arithmetic Coding with the estimated pmf of the source.\n"3" for Golomb Coding with fixed m.\n"4" for slow adaptive Golomb-Rice coding (exhaustive search of k).\n"5" for Exp-Golomb coding.\n"6" for fast adaptive Golomb-Rice coding.\n ';
choice = input(prompt);
while choice <=0 || choice > 6
    fprintf('\nPlease select a correct value as mentioned...')
    choice = input(prompt) ;
end













%% Linear Predictive Model

%Total number of samples and block length
total_samples = length(audio_normalized);
block_length = 1024;
total_blocks= floor(total_samples/block_length);
blocks = zeros(total_blocks, block_length);

%Framing process for the input audio data
for i = 1:total_blocks
    for j = 1:block_length
        blocks(i,j) = audio_normalized((i-1)*block_length + j);
    end
end
blocks=blocks' ;

%The Linear Predictive Model is implemented based on integer operation. 
%The transmission of the a coefficients is accomplished by using Parcor coefficients ( pk , k = 1,…, lpc_order), which can be obtained by using the Levinson-Durbin algorithm.
prompt2='\nDo you want to plot the mean squared error function for several orders? If yes type "1" ';
x2=input(prompt2);
if(x2==1)
   plotMSE(blocks);
end
tic
lpc_order = 60; 
a_coefs = zeros(lpc_order+1,total_blocks);
k_coefs = zeros(lpc_order,total_blocks);


%The Levinson-Durbin Algorithm
for i = 1:total_blocks
    [Autocor_Func,lags] = autocorr(blocks(:,i),'NumLags',lpc_order); %sample autocorrelation function
    [a,E,kref] = Levinson_Durbin(Autocor_Func,lpc_order); % the Levinson-Durbin Algorithm
    k_coefs(:,i)= kref;
    a_coefs(:,i) = a;
end


%%%%%%%%%%%%%Quantization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We quantize k_coefs so that the quantized values, 'quant_k', are 
% inside the range [-64,63].
quant_k=ParCor_Quantization(lpc_order,total_blocks,k_coefs);
%%%%%%%%%%%%De-Quantization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dequant_k=ParCor_Dequantization(quant_k,lpc_order,total_blocks);
%-------------------------k-coefficients to a-coefficients -----------
%  Conversion of reconstructed Parcor coefficients into direct c coefficients
%  This is the reconstructed_signal procedure that is similar to the k-parameters to
%  a-coefficients algorithm as shown in the Discrete-time Signal Processing
%  book by A.Oppenheim and R.Schaffer at page 410
%  The reconstructed Parcor coefficients are converted to k-order 
% (1 < j < lpc_order) a coefficients. 
% 'a_coef' : Linear Prediction Coefficients

a_coef = k_to_a_coefficients(dequant_k,total_blocks,lpc_order);

% %----------------------------LINEAR PREDICTIVE MODELING----------------------
% The Linear Predictor generates a prediction for each sample in the frame.
% After that, it computes the prediction prediction_errors.
% 'prediction_errors': Prediction prediction_errors----->INTRA FRAME LINEAR PREDICTOR

prediction_errors = Linear_Prediction_Error(lpc_order,blocks,a_coef);

perrors1d = reshape(prediction_errors.',1,[]);
figure;
histogram(perrors1d/(2^15),'Normalization','probability')
xlabel('Prediction error')
ylabel('Probability')
title('Histogram for the prediction error signal')



original_entropy=calculate_entropy(perrors1d);
sprintf('Shannon entropy of the non-flattened source is %f bits/symbol',original_entropy)


%% (2) Pre-processor

total_blocks = length(prediction_errors(1,:));
block_length = length(prediction_errors(:,1));
lpc_order = length(quant_k(:,1));
L = min(lpc_order,16);

%Store the signs of the prediction errors
signs = sign(prediction_errors);
%2.1 
% NUMBER OF SHIFTS ->shift(:,:)
% The number of down-shifts applied to each prediction error. The
% number of shifts can be calculated from the quantized_k coefficients.
% They actually are the bits that we have to decrease from the
% prediction errors from each frame.


shift = zeros(L,total_blocks);

s1 = 0;
s2 = 0;

for i = 1:total_blocks
    for j = 1:2
        for u = 1:j
            s1 = s1 + RA_shift12(RA_shift12(:,1) == quant_k(u,i),2);
        end
        shift(j,i) = floor((2.^12 + s1)/2.^13); %store shift(j,i) for samples 1 and 2
        s1 = 0;
    end

    for j = 3:L
        for u = 1:2
            s1 = s1 + RA_shift12(RA_shift12(:,1) == quant_k(u,i),2);
        end
        
        for u = 3:j
            s2 = s2 + RA_shift(RA_shift(:,1) == abs(quant_k(u,i)),2);
        end
        
        shift(j,i) = floor((2.^12 + s1 + s2)/2.^13); %store shift(j,i) for samples 3 to L
        s1 = 0;
        s2 = 0;
    end
end

%2.2 DOWN-SHIFT OPERATION

% LSB data   

LSB = zeros(L,total_blocks);

for i = 1:total_blocks
    for j = 1:L
        LSB(j,i) = mod(abs(prediction_errors(j,i)), pow2(shift(j,i)));
    end
end
%2.3
%FLATENED PREDICTION ERRORS
% In the standard we don't have information about the sign of the signal but
% the flat_errors contain information about the sign of the signal. 
% That is because some floor operations in LSB computing loses the sign due to the value is 0.

flat_errors = zeros(block_length,total_blocks);

for i = 1:total_blocks
    for j = 1:L
        dec = ceil(abs(prediction_errors(j,i)));
        % bitshift(A,k) returns A shifted to the left by k bits, equivalent 
        % to multiplying by 2^k. Negative values of k correspond to shifting 
        % bits right or dividing by 2^|k| and rounding to the nearest integer 
        % towards negative infinity.
        flat_errors(j,i) = bitshift(dec, -shift(j,i));
    end
    
    for j = L+1:block_length
        flat_errors(j,i) = abs(prediction_errors(j,i));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
perrors1d2 = reshape(flat_errors.',1,[]);
flat_entropy=calculate_entropy(perrors1d2);
sprintf('Shannon entropy of the flattened source is %f bits/symbol ',flat_entropy)

figure;
histogram(perrors1d2/(2^15),'Normalization','probability')
xlabel('Flattened Prediction Error')
ylabel('Probability')
title('Histogram for the flattened residual')

%% (3) Entropy Encoder

 [~, fn, ~] = fileparts(pcm_input);
 codedFile = ['.\' fn '_encoded'];
 fileID = fopen(codedFile,'w');
if choice~=4 && choice~=5 && choice~=6
    positive_flat=flat_errors+1;
    [r, c]=size(positive_flat);
    s=1:max(max(positive_flat))+1;
    estimated_counts= round(r*c*gaussmf(s,[0.3*max(s),-0.1])); 
    % encoded_bitstream = cell(1,total_blocks);
    %bitstreamlength=zeros(1,total_blocks);
    golomb_index=0;
    meanOfBlock=zeros(1,total_blocks);
    EOF1=max(max(positive_flat))+1 ;
    if choice==3
        prompt3='Choose m:\n';
        m=input(prompt3);
    end
    for i=1:total_blocks
       if choice==1
           sequence=positive_flat(:,i)';
           %Make a mapping
           useq=unique(sequence);
           U(i,:)=[useq,zeros(1,length(sequence)-length(useq))];
           counts=countmember(useq,sequence);
           countz(i,:)=[counts zeros(1,length(sequence)-length(counts))];
           [found, idx]=ismember(sequence,useq);
           %Encoding procedure
           encoded_bitstream{i} = arithmetic_encoder(idx, counts);       
           bitstreamlength(i) = length(encoded_bitstream{i});
           fwrite(fileID, encoded_bitstream{i},'ubit1');
       elseif choice==2
           sequence=positive_flat;
           %EOF2=max(sequence(:,i)')+1;
           encoded_bitstream{i} = arithmetic_encoder(sequence(:,i)', estimated_counts);       
           bitstreamlength(i) = length(encoded_bitstream{i});
           fwrite(fileID, encoded_bitstream{i},'ubit1');
       elseif choice==3
           
           sequence=flat_errors;
           meanOfBlock(i)=mean(blocks(:,i));
           for j2=1:block_length
             golomb_index=golomb_index+1;
             encoded_bitstream{golomb_index} = GolombEncoder(sequence(j2,i),m); %m can be an arbitrary integer
             bitstreamlength(golomb_index) = length(encoded_bitstream{golomb_index});
             fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
           end
       end
    end
else
    if(choice==4)
         sequence=flat_errors;
         [encoded_bitstream, m, bitstreamlength] = GolombRiceEncoderFunc(sequence, fileID);
    elseif(choice==5)
        sequence=flat_errors;
        [encoded_bitstream, k, bitstreamlength] = expGolombEncoderFunc(sequence, fileID);
    elseif(choice==6)
        golomb_index=0;
        sequence=flat_errors;
        k = floor(log2(mean(abs(perrors1d2)))); %Absolute Average Guess like in MPEG-4 ALS, ECG telemedicine applications etc
        m=2^k;
        for i=1:total_blocks
            for j2=1:block_length
                golomb_index=golomb_index+1;
                encoded_bitstream{golomb_index} = GolombRiceEncoder(sequence(j2,i),m);
                bitstreamlength(golomb_index) = length(encoded_bitstream{golomb_index});
                fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
            end
        end
    else
        fprintf("Other non-defined source coding technique")
    end
end
meanLen=(sum(bitstreamlength))/(total_blocks*block_length)
Redundancy=((meanLen-original_entropy)/original_entropy)*100

toc
%% Find Bytes of coded file and then Compression Ratio
dirCodedFile=dir(codedFile);
codedFileBytes=dirCodedFile.bytes;

fclose(fileID);

dirInpF=dir(pcm_input);
codedInpBytes=dirInpF.bytes;


if (Columns~=2)
    [rq,cq] = size(quant_k);
    [rL,cL] = size(LSB);
    [rS,cS] = size(signs);
    compressionRatio = codedInpBytes/(codedFileBytes + (7*rq*cq + nextpow2(max(max(shift)))*rL*cL + rS*cS )/8)
else
    [rq,cq] = size(quant_k);
    [rL,cL] = size(LSB);
    [rS,cS] = size(signs);
    [rSide, cSide] = size(Side);
    compressionRatio = codedInpBytes/(codedFileBytes + ( 7*rq*cq + (nextpow2(max(max(shift))))*rL*cL + rS*cS + rSide*cSide*16 )/8)
end



t_input=1/Fs:1/Fs:length(audio_normalized)/Fs;
t1= 1/Fs:1/Fs:length(perrors1d)/Fs;
t2= 1/Fs:1/Fs:length(perrors1d2)/Fs;
figure;
hold on
plot(t_input,audio_normalized/(2^15));
plot(t1,perrors1d/(2^15));
plot(t2,perrors1d2/(2^15));
legend('Audio Input','Residual','Flattened Residual')
xlabel('Duration (seconds)');
ylabel('Amplitude');


if choice ==4 || choice==6
    if Columns~=2
        clearvars -except quant_k encoded_bitstream LSB signs block_length total_blocks lpc_order m choice codedFile flat_errors Columns Fs
        
    else
       clearvars -except quant_k encoded_bitstream LSB signs block_length total_blocks lpc_order m choice codedFile flat_errors Columns Side Fs
    end
elseif choice==5
    if Columns ~=2
        clearvars -except quant_k encoded_bitstream LSB signs block_length lpc_order k choice codedFile flat_errors Columns Fs
    else
        clearvars -except quant_k encoded_bitstream LSB signs block_length lpc_order k choice codedFile flat_errors Columns Side Fs
    end
end
%lpc_order, block_length, total_blocks, k, choice, Fs will be stored only one time ---> very few bits of storage
%All the others must be stored for every block

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%SEPARATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%OF%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%ENCODER%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%AND%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%DECODER%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (4) Entropy Decoder Step (has input the encoded_bitstream)
tic
fileID = fopen(codedFile, 'r');

golomb_index=0;
if choice==1
    total_blocks=length(encoded_bitstream(1,:));
    decoded_sequence = zeros(block_length, total_blocks);
    decflat_errors = zeros(block_length, total_blocks);
elseif choice==2
    inpt = encoded_bitstream;
    estimated_counts= round(r*c*gaussmf(s,[0.3*max(s),-0.1])); 
    decoded_sequence = zeros(block_length, total_blocks);
    decflat_errors = zeros(block_length, total_blocks);
elseif choice==3
    total_blocks=(length(encoded_bitstream(1,:)))/block_length;
    decoded_sequence = zeros(block_length, total_blocks);
elseif choice==4 || choice==5 || choice==6
    total_blocks=(length(encoded_bitstream(1,:)))/block_length;
    inpt=fread(fileID,'ubit1');
    inpt=inpt';
end


for i=1:total_blocks
        if choice==1 %Non-realistic way
            countzz=countz(i,:); countzz=countzz(countzz>0);
            inpt = fread(fileID, bitstreamlength(i),'ubit1');
            decoded_sequence(:,i)= arithmetic_decoder(inpt', countzz, block_length);
            Aid=decoded_sequence(:,i);
            B=U(i,:);
            B=B(B>0);
            decoded_sequence(:,i)=B(Aid)';
        elseif choice==2 %Semi-Realistic way
            decoded_sequence(:,i) = arithmetic_decoder(inpt{1,i}', estimated_counts, block_length);
        elseif choice==3 %Non-realistic way yet
            
            for j=1:block_length
                golomb_index=golomb_index+1;
                inpt = fread(fileID, bitstreamlength(golomb_index),'ubit1');
                decoded_sequence(j,i) = GolombDecoder(inpt',m);
            end 
        end
        if choice==1 || choice==2
            % Undo avoid zero values by adding 1
            decflat_errors(:,i) = decoded_sequence(:,i) - 1 ;
        elseif choice==3
            decflat_errors(:,i)= decoded_sequence(:,i);
        end
end

if choice==4 || choice ==6 %Realistic ways
	size(inpt)
	decoded_sequence=GolombRiceDecoder(block_length,total_blocks,inpt,m);
    decoded_sequence=reshape(decoded_sequence,block_length,[]);
    decflat_errors=decoded_sequence;
    m
elseif choice==5
    size(inpt)
    k
    decoded_sequence=exp_Golomb_Dec(inpt,k,block_length,total_blocks);
    decoded_sequence=reshape(decoded_sequence,block_length,[]);
    decflat_errors=decoded_sequence;
     
end




fprintf('\nChecking for decoding validity...\n');
 % Check that decoded_sequence matches the original seq.
  if isequal(flat_errors,decflat_errors)
     fprintf('Decoding successful\n');
  end

%% -(5) Post-processor step-
total_blocks = length(decflat_errors(1,:));
lpc_order = length(quant_k(:,1));
block_length = length(decflat_errors(:,1));

%---------------------- NUMBER OF SHIFTS ----------------------------------
% 'shift': Number of down-shifts applied to each residue.
%          Shifts number are calculated from the quantized PARCOR
%          coefficients (Bits that we have to decrease from the
%          prediction residues from each frame).

L = min(lpc_order,16);
shift = zeros(L,total_blocks);
sum1 = 0;
sum2 = 0;
RA_shift12 = dlmread('RA_shift12.txt');   % RA_shift12 table
RA_shift = dlmread('RA_shift.txt');   % RA_shift table

for i = 1:total_blocks
    for j = 1:2
        for k = 1:j
            sum1 = sum1 + RA_shift12(RA_shift12(:,1) == quant_k(k,i),2);
        end
        shift(j,i) = floor((2.^12 + sum1)/2.^13);
        sum1 = 0;
    end

    for j = 3:L
        for k = 1:2
            sum1 = sum1 + RA_shift12(RA_shift12(:,1) == quant_k(k,i),2);
        end
        
        for k = 3:j
            sum2 = sum2 + RA_shift(RA_shift(:,1) == abs(quant_k(k,i)),2);
        end
        
        shift(j,i) = floor((2.^12 + sum1 + sum2)/2.^13);
        sum1 = 0;
        sum2 = 0;
    end
end

post_residues = zeros(block_length, total_blocks);

for i = 1:total_blocks
    for j = 1:L
        post_residues(j,i) = decflat_errors(j,i) * pow2(shift(j,i)) + LSB(j,i);
        
        if (signs(j,i) == -1)   % Correction of sign value
            post_residues(j,i) = - post_residues(j,i);
        end
    end
       
    for j = L+1:block_length
        post_residues(j,i) = decflat_errors(j,i);
        
        %Using entropy coders
        if (signs(j,i) == -1)   % Correction of sign value
            post_residues(j,i) = - post_residues(j,i);
        end
    end 
end 
  
%% -(6) reconstructed_signal step-

%   'total_blocks': Number of frames processed

total_blocks = length(quant_k(1,:));
lpc_order = length(quant_k(:,1));
block_length = length(post_residues(:,1));
n_samples = total_blocks * block_length;

%----------------------------PARCOR DE-QUANTIZATION------------------------
% 'dequant_k': De-quantizated PARCOR coefficients

dequant_k = zeros(lpc_order, total_blocks);

for i = 1:total_blocks
    dequant_k(1,i) = (2 * ((exp(quant_k(1,i)/64*log(3/2))-(2/3)) * 6/5).^2)-1; 
    dequant_k(2,i) = -(2 * ((exp(quant_k(2,i)/64*log(3/2))-(2/3)) * 6/5).^2)+1;
    dequant_k(3:lpc_order,i) = quant_k(3:lpc_order,i)/64; 
end

%----------------------------PARCOR TO LPC---------------------------------
% The reconstructed PARCOR coefficients are converted to k-order 
% (1 < j < lpc_order) LPC coefficients lpc(j,1..j) 
% 'lpc' : Linear Prediction Coefficients 
a_coef = k_to_a_coefficients(dequant_k,total_blocks,lpc_order);
%-------------------------LINEAR PREDICTOR DECODER-------------------------
% The Linear Predictor Decoder reconstructs the audio input signal from the
% prediction post_residues of each sample in the frame.
% 'audio_ouput': audio_ouput of prediction post_residues
[ro, co]=size(decflat_errors);
reconstructed_signal = Linear_Prediction_Reconstruction(post_residues,a_coef,block_length,total_blocks,lpc_order);
  

audio_output = reshape(reconstructed_signal/2^16, 1, ro*co);
if Columns==2
	%CHANNEL CORRELATION
	L=audio_output+((Side(1:length(audio_output)))')./2;
	R=audio_output-((Side(1:length(audio_output)))')./2;
	audio_output=[L R];
end
toc

soundsc(audio_output, Fs);
audiowrite('.\audio_output.wav',audio_output,Fs);


% y=blocks(:,total_blocks);
% for i=1:lpc_order+1
%     r(i)=0;
%     for j=i+1:length(blocks(:,total_blocks))
%         r(i) = y(j)*y(j-i) + r(i);
%         
%     end   
%     autoC(i) = mean(r(i)) / (length(blocks(:,total_blocks))-1);
% end