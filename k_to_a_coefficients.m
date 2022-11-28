function a = k_to_a_coefficients(k,M,order)
%  This is the reconstructed_signal procedure that is identical to the k-parameters to
%  a-coefficients algorithm as shown in the Discrete-time Signal Processing
%  book by A.Oppenheim and R.Schaffer at page 410

a = zeros(order, order, M);
for n = 1:M
	for i = 1:order
    	a(i,i,n) = k(i,n);
        if i>1
            for j = 1:i-1
            	a(j,i,n) = a(j,i-1,n) + k(i,n)*a(i-j,i-1,n);
            end
        end
	end
end


end

