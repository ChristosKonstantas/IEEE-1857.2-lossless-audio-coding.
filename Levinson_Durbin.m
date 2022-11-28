function[a,E,k]=Levinson_Durbin(R, prediction_order)
%% INIT

k=zeros(prediction_order,1); E=zeros(prediction_order,1);
a_0=1; E_0=R(1);  k_0=-R(2)/R(1); 


%In matrix a, every row represents a different order
%In every row we will have a result until the value 1 of the a's eye matrix
%These are the e^(i-1) matrices, that we calculate every time
%The last row of the matrix has our prediction order a-coefficients (we start counting from 0)
%At the end of the algorithm the last row flipped will represent a_1,a_2,...,a_{order}

a=eye(prediction_order+1,prediction_order+1);
%First iteration results to simplify the Levinson-Durbin Algorithm
  
k(1)=-R(2)/R(1); 
a(2,1)=k(1);
E(1)=E_0*(1-k(1)^2); 
i=1;
%% LEVINSON-DURBIN
while 1   
    if i==prediction_order
        break;
    else  
        m=i+1;
        %Find PARCOR coefficient k and store it to the k-coefficients array:
        k(m) = -(a(m,1:m)*R(2:m+1))/E(i); % instead of sum we do multiplication with a 1xA and an Ax1 matrix so we have an 1x1 result ( a(j,1:j) is 1xA and R(2:j+1) is Ax1 )
        %Store the a_coefficient with the greatest order to the first column of row m+1, of array a():
        a(m+1,1) = k(m); % store the result above to the first column of the a-coefficients eye matrix, that is actually the last value (the last order is for the first column)
        %Then for the rest of elements of a(), store the result:
        a(m+1,2:m)=a(m,1:m-1)+k(m)* fliplr(a(m,1:m-1));
        %
        E(m) =E(i)*(1-k(m)^2); 
        i=i+1;
    end
end

a =  fliplr(a(prediction_order+1,1:prediction_order)); % Take the last row and flip it (last row has the last order)
a=[a_0,a];




end