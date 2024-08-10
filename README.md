# IEEE-1857.2-lossless-audio-coding.
*It works very well, but to do it in realistic scenarios we must consider C++ implementation of the IEEE 1857.2 encoder and decoder as long as the functions that are being used. If you would like to participate
in this you can directly communicate with me. Same techniques can be used for ECG signal lossless compression.*

Run IEEE_1857dot2.m file from your directory.

In this code we implement the IEEE 1857.2 lossless audio coding extension using 5 different source coding techniques: 
1) Arithmetic Coding using the actual probability mass function of the source (benchmark).
2) Arithmetic Coding using an estimated probability mass function of the source.
3) Golomb Coding choosing m.
4) Adaptive Golomb-Rice coding with exhaustive search for the best k where m=2^k -> slow encoding fast decoding.
5) Adaptive Exponential Golomb-coding with exhaustive search for the best k -> slow encoding fast decoding.
6) Adaptive Golomb-Rice coding -> fast encoding fast decoding. 

If you want to process the results and understand deeper what is going on, the analytical report of this code is my thesis: see THESIS report.pdf file!


We apply the linear prediction model with fixed block size on the digital audio signal and then its probabilities are exploited by the source coding techniques.


Execute: IEEE_1857dot2.m

---

**General implementation**

![image](https://github.com/user-attachments/assets/adac4221-f27f-46d0-a145-6fbd6f4dfef4)


**Linear predictive modeling (Analysis/Synthesis)**
![image](https://github.com/user-attachments/assets/5d798604-fbe6-49f0-b7c6-80056384759f)

---

**IEEE-1857.2 Encoder**
![image](https://github.com/user-attachments/assets/28c4332f-062c-4f83-bee5-327cf28ab6a0)


**IEEE-1857.2 Decoder**
![image](https://github.com/user-attachments/assets/f930f378-164a-4e1a-a4d1-32ea6adf0e6c)

---

**Results**


**More information**

This [thesis](https://github.com/ChristosKonstantas/IEEE-1857.2-lossless-audio-coding./blob/main/THESIS%20report.pdf) provides a comprehensive guide to lossless audio compression and coding. It includes a MATLAB-based implementation that emulates the [IEEE 1857.2 lossless audio coding extension](https://standards.ieee.org/ieee/1857.2/10714/) offering capabilites of experimenting with new source encoders/decoders. Also, to improve compression efficiency we may consider adding [ATC-ABS](https://ieeexplore.ieee.org/abstract/document/116122) method.
