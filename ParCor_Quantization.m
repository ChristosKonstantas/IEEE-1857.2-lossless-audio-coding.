function quant_parcor = ParCor_Quantization(lpc_order,total_blocks,k_coefs)
%PARCOR COMPANDING QUANTIZATION

quant_parcor = zeros(lpc_order,total_blocks);
for i = 1:total_blocks
    quant_parcor(1,i) = floor(64*log(2/3 + 5/6*sqrt((1+k_coefs(1,i))/2))/log(3/2)); 
    quant_parcor(2,i) = floor(64*log(2/3 + 5/6*sqrt((1-k_coefs(2,i))/2))/log(3/2));
    quant_parcor(3:lpc_order,i) = floor(64*k_coefs(3:lpc_order,i));
end

end

