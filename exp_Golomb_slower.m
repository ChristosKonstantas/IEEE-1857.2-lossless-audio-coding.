function codeword = exp_Golomb_slower(n,k)
q1=floor(log2(1+n/(2^k)));
q=ones(1,q1);
q=[q,0]; %Unary Coding for quotient
su=0;
q2=q1;
for j1=0:q1-1
    su=su+2^(j1+k);
end
r=n-su;
r_code=de2bi(r,k+q2,'left-msb'); %k+q2-bit binary code of r
codeword=[q r_code];
end

