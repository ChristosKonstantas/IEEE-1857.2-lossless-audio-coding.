function [L,R] = Channel_Correlation(Side,Mid)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
	L=Mid+((Side(1:length(Mid)))')./2;
	R=Mid-((Side(1:length(Mid)))')./2;
end

