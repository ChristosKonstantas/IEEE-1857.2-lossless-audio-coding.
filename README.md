# IEEE-1857.2-lossless-audio-coding.

Run IEEE_1857dot2.m file from your directory.

In this code we implement the IEEE 1857.2 lossless audio coding extension using 5 different source coding techniques: 
1) Arithmetic Coding using the actual probability mass function of the source (benchmark).
2) Arithmetic Coding using an estimated probability mass function of the source.
3) Golomb Coding choosing m.
4) Adaptive Golomb-Rice coding with exhaustive search for the best k where m=2^k -> slow encoding fast decoding.
5) Adaptive Exponential Golomb-coding with exhaustive search for the best k -> slow encoding fast decoding.
6) Adaptive Golomb-Rice coding -> fast encoding fast decoding. 

If you want to process the results and understand deeper what is going on, the analytical report of this code is my thesis: see THESIS report.pdf file!

Thank you very much for studying this code!
