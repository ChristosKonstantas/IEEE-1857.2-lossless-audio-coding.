function [Mid,Side] = Channel_Decorrelation(audio_signal)
%STEREO ONLY
    L=audio_signal(:,1); R=audio_signal(:,2);
	Mid=(L+R)/2;
	Side=(L-R);
end

