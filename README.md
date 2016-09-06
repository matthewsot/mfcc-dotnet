# mfcc-dotnet
An FSharp/.NET library for MFCC audio feature extraction.

# Implementation
This MFCC implementation is partially based off of the guide at [Practical Cryptography](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank).

The specific steps it takes are:
1. Compute a Hamming window using MathDotNet and apply it to the samples
2. Perform an in-place FFT on the windowed samples using MathDotNet, then take the magnitude of each of the FFT outputs
3. Compute the MFCC filterbank (or read from cached version)
4. Apply the filterbank
5. Log the outputs
6. Run a DCT on the logged outputs

# Performance
I have tried to keep optimization in mind, but no promises are made regarding the speed or efficiency of the code.

Some optimizations include:
- The filterbank is cached after it's first created
- Filters are applied and summed as one, instead of applied separately and summed afterwards
- You can uncomment a line to simply sum the real and imaginary FFT outputs instead of taking the absolute value of the sum of squares, which will result in a small performance increase (though might impact accuracy).

PRs are also very much appreciated if you spot anything else that should be optimized!