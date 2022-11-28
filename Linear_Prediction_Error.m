function er = Linear_Prediction_Error(order,blocks,a)

[len , total] = size(blocks);
er = zeros(order,total);

for i = 1:total
    er(1,i) = blocks(1,i); %e[0]=
    
    for n = 2:order
        s = 0;
        for k = 1:n-1
            s = s + a(k,n,i)*blocks((n-k),i);
        end 
        yhat=-round(s);
        er(n,i) = blocks(n,i)-yhat;
    end
   
    for n = order+1:len
        s = 0;
        for k = 1:order
            s = s + a(k,order,i)*blocks(n-k,i);
        end 
        yhat=-round(s);
        er(n,i) = blocks(n,i)-yhat;
    end
end

end

