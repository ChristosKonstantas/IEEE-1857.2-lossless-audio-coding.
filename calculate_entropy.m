function H = calculate_entropy(X)
pmf = histcounts(X,[unique(X) Inf],'Normalization','probability');
pmf=pmf';
s=0;
for i=1:length(pmf)
    s=s-pmf(i)*log(pmf(i));
end
H=s;

end

