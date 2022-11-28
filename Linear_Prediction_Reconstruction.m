function y = Linear_Prediction_Reconstruction(e,a,len,total,order)
y = zeros(len, total);

for i = 1:total
    y(1,i) = e(1,i);
    
    for j = 2:order
        s = 0;
        for k = 1:j-1
            s = s + a(k,j,i)*y((j-k),i);
        end
        yhat=-round(s);
        y(j,i) = e(j,i)+yhat;
    end
   
    for j = order+1:len
        s = 0;
        for k = 1:order
            s = s + a(k,order,i)*y(j-k,i);
        end 
        yhat=-round(s);
        y(j,i) = e(j,i)+yhat;
    end
end
end

