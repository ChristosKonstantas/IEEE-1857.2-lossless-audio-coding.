clear all; close all; clc;
%% Initialization

%Load all the important files for the standard (if it doesn't work please insert your directory or add your path!)
Gamma = load('.\Gamma.txt');% Compound Quantization function
RA_shift12 = dlmread('.\RA_shift12.txt');   % RA_shift12 table
RA_shift = dlmread('.\RA_shift.txt');   % RA_shift table

pcm_input='.\handel.wav'; %Choose the directory of the audio file that you want to compress


yy=0;
for lpc_order=2:1023
    yy=yy+1;
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
        audio_normalized=Mid*(2^15); %the input audio_signal is 16-bit, little-endian PCM represented in fractional so it needs to be shifted left 16 times
    else
        fprintf('No code yet for more than 2 channels')
    end


    
    choice = 4;
    while choice <=0 || choice > 5
        fprintf('\nPlease select a correct value as mentioned...')
        choice = input(prompt) ;
    end

    %Total number of samples and block length
    total_samples = length(audio_normalized);
    block_length = 2048;
    total_blocks= floor(total_samples/block_length);
    blocks = zeros(total_blocks, block_length);

    %Blocking the input audio data
    for i = 1:total_blocks
        for j = 1:block_length
            blocks(i,j) = audio_normalized((i-1)*block_length + j);
        end
    end
    blocks=blocks' ;











    %% Linear Predictive Modeling

    %The Linear Predictive Model is implemented based on integer operation. 
    %The transmission of the a coefficients is accomplished by using Parcor coefficients ( pk , k = 1,…, lpc_order), which can be obtained by using the Levinson-Durbin algorithm.
%     prompt2='\nDo you want to plot the mean squared error function for several orders? If yes type "1" ';
%     x2=input(prompt2);
%     if(x2==1)
%        plotMSE(blocks);
%     end
    tic
    a_coefs = zeros(lpc_order+1,total_blocks);
    k_coefs = zeros(lpc_order,total_blocks);


    %The Levinson-Durbin Algorithm
    for i = 1:total_blocks
        [Autocor_Func,lags] = autocorr(blocks(:,i),'NumLags',lpc_order); %sample autocorrelation function
        [a,e,kref] = Levinson_Durbin(Autocor_Func,lpc_order); % the Levinson-Durbin Algorithm
        k_coefs(:,i)= kref;
        a_coefs(:,i) = a;
    end

    perrors1d=reshape(e.',1,[]);
    MSE_before=mean(perrors1d.^2);
    %----------------------------PARCOR QUANTIZATION---------------------------
    % Parcor coefficients (k_coef, k = 1,...,lpc_order) are first quantized by 
    % the companding function. The resulting quantized values, 'quant_k', are 
    % restricted to the range [-64,63].
    quant_k=ParCor_Quantization(lpc_order,total_blocks,k_coefs);
    % 
    % %----------------------------PARCOR DE-QUANTIZATION------------------------
    % % 'dequant_k': De-quantizated PARCOR coefficients
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
    % With linear prediction we generate a prediction for each sample in the block and then we compute the prediction_errors.
    prediction_errors = Linear_Prediction_Error(lpc_order,blocks,a_coef);
    %1-D prediction errors
    perrors1d = reshape(prediction_errors.',1,[]);
    %Save signs of prediction errors
    signs = sign(prediction_errors);

    original_entropy=calculate_entropy(perrors1d);
    sprintf('Shannon entropy of the non-flattened source is %f bits/symbol',original_entropy)


    %% (2) Pre-processor

    total_blocks = length(prediction_errors(1,:));
    block_length = length(prediction_errors(:,1));
    lpc_order = length(quant_k(:,1));
    L = min(lpc_order,16);


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
            shift(j,i) = floor((2.^12 + s1)/2.^13);
            s1 = 0;
        end

        for j = 3:L
            for u = 1:2
                s1 = s1 + RA_shift12(RA_shift12(:,1) == quant_k(u,i),2);
            end

            for u = 3:j
                s2 = s2 + RA_shift(RA_shift(:,1) == abs(quant_k(u,i)),2);
            end

            shift(j,i) = floor((2.^12 + s1 + s2)/2.^13);
            s1 = 0;
            s2 = 0;
        end
    end

    %2.2
    %DOWN-SHIFT OPERATION

    % LSB is the sign of the signal (Matlab LSB = Real MSB)   

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

    %% (3) Entropy Encoder

    [~, fn, ~] = fileparts(pcm_input);
    codedFile = ['C:\Users\titom\Desktop\Diplomatikh\matlab\final\' fn '_encoded'];
    fileID = fopen(codedFile,'w');

    positive_flat=flat_errors+1;
    [r, c]=size(positive_flat);
    s=1:max(max(positive_flat))+1;
    estimated_counts= round(r*c*gaussmf(s,[0.3*max(s),-0.1])); 

    % encoded_bitstream = cell(1,total_blocks);
    %bitstreamlength=zeros(1,total_blocks);
    golomb_index=0;
    m=2^10;
    meanOfBlock=zeros(1,total_blocks);
    EOF1=max(max(positive_flat))+1 ;
    for i=1:total_blocks
       if choice==1
           %Append EOF symbol
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
             encoded_bitstream{golomb_index} = golomb_enco(sequence(j2,i),m); %m can be an arbitrary integer
             bitstreamlength(golomb_index) = length(encoded_bitstream{golomb_index});
             fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
           end
       elseif choice==4
           sequence=flat_errors;
            for j2=1:block_length
                golomb_index=golomb_index+1;
                encoded_bitstream{golomb_index} = GolombRiceEncoder(sequence(j2,i),m); %m=2^k
                bitstreamlength(golomb_index) = length(encoded_bitstream{golomb_index});
                fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
            end
       elseif choice==5
           sequence=flat_errors;
           k=10;
           for j=1:block_length
            golomb_index=golomb_index+1;
            n=sequence(j,i);
            encoded_bitstream{golomb_index}=exp_Golomb(sequence(j,i),k);
            bitstreamlength(golomb_index) = length(encoded_bitstream{golomb_index});
            fwrite(fileID, encoded_bitstream{golomb_index},'ubit1');
           end
       end

    end
    meanLen=(sum(bitstreamlength))/(total_blocks*block_length)
    Redundancy=((meanLen-original_entropy)/original_entropy)*100


    %% Find Bytes of coded file and then Compression Ratio
    
    %LSB needs max 2bits for each value
    %quant_k needs max 7bits for each value
    %signs needs max 1 bit for each value
    [rq cq] = size(quant_k); %7
    [rl cl] = size(LSB); %2
    [rs cs] = size(signs); %1

    dirCodedFile=dir(codedFile);
    codedFileBytes=dirCodedFile.bytes;

    fclose(fileID);

    dirInpF=dir(pcm_input);
    codedInpBytes=dirInpF.bytes;
    compressionRatio(yy)=codedInpBytes/(codedFileBytes + (7*rq*cq)/8 + (2*rl*cl)/8 + (rs*cs)/8 )
    
    

end

t=1:yy-1;
t=t+1;

figure;
stem(t,compressionRatio)
ylabel('Compression Ratio')
xlabel('Prediction Order')



