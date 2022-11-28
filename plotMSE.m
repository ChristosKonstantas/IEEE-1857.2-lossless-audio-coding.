function plotMSE(blocks)

[block_length,total_blocks]=size(blocks);

%Below you can select the maximum order you want to have in the plot
max_order=200;
for lpc_order=1:max_order
    a_coefs = zeros(lpc_order+1,total_blocks);
    parcor_coefs = zeros(lpc_order,total_blocks);
    %The Levinson-Durbin Algorithm
    for i = 1:total_blocks
        [Autocor_Func,lags] = autocorr(blocks(:,i),'NumLags',lpc_order); %sample autocorrelation function
        [a,e,k] = Levinson_Durbin(Autocor_Func,lpc_order); % the Levinson-Durbin Algorithm
        a_coefs(:,i) = a;
        parcor_coefs(:,i)= k;  
    end
    MSE(lpc_order)=mean(e.^2);
end
order=1:max_order;
figure;
plot(order,MSE)
title('Mean squared error vs Prediction order')
xlabel('Linear prediction order')
ylabel('MSE')