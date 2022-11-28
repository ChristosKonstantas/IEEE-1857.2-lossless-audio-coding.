function audio = Reconstruction(quant_k,post_residues,decflat_errors)
% 'total_blocks': Number of blocks processed
[lpc_order,total_blocks] = size(quant_k);
block_length = length(post_residues(:,1));

%----------------------------PARCOR DE-QUANTIZATION------------------------
% 'dequant_k': De-quantizated PARCOR coefficients
dequant_k=ParCor_Dequantization(quant_k,lpc_order,total_blocks);
%----------------------------PARCOR TO LPC---------------------------------
% The reconstructed PARCOR coefficients are converted to k-order 
% (1 < j < lpc_order) LPC coefficients lpc(j,1..j) 
% 'lpc' : Linear Prediction Coefficients 
a_coef = k_to_a_coefficients(dequant_k,total_blocks,lpc_order);
%-------------------------LINEAR PREDICTION DECODER-------------------------
% The Linear Predictor Decoder reconstructs the audio input signal from the
% prediction post_residues of each sample in the frame.
% 'audio_ouput': audio_ouput of prediction post_residues
[ro, co]=size(decflat_errors);
reconstructed_signal = Linear_Prediction_Reconstruction(post_residues,a_coef,block_length,total_blocks,lpc_order);
audio = reshape(reconstructed_signal/2^15, 1, ro*co);


end

