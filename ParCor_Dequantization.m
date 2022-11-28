function dequant_k = ParCor_Dequantization(quant_k,lpc_order,total_blocks)
%PARCOR COMPANDING DEQUANTIZATION
dequant_k=zeros(lpc_order,total_blocks);
% for i = 1:total_blocks
%     dequant_k(1,i) = Gamma(Gamma(:,1)==quant_k(1,i),2); 
%     dequant_k(2,i) = -Gamma(Gamma(:,1)==quant_k(2,i),2);
%     dequant_k(3:lpc_order,i) = quant_k(3:lpc_order,i)*(2^14) + 2^13; 
% end
for i = 1:total_blocks
    dequant_k(1,i) = (2*((exp(quant_k(1,i)/64*log(3/2))-(2/3)) * 6/5).^2)-1; 
    dequant_k(2,i) = -(2*((exp(quant_k(2,i)/64*log(3/2))-(2/3)) * 6/5).^2)+1;
    dequant_k(3:lpc_order,i) = quant_k(3:lpc_order,i)/64; 
end


end

