function codeword = exp_Golomb(n,k)
q1=floor(log2(1+n/(2^k)));
q=ones(1,q1);
q=[q,0]; %Unary Coding for quotient
su=0;
r=n-(2^k)*((2^q1)-1);
r_code=de2bi(r,k+q1,'left-msb'); %k+q2-bit binary code of r
codeword=[q r_code];
end

