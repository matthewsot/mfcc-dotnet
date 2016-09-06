namespace MFCCDotNet

open System.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.IntegralTransforms
open MathNet.Numerics

module public MFCC =
    type triangle_filter = { start_bin: int; peak_bin: int; end_bin: int; }

    let mel (frequency) =
        1125.0 * log10 (1.0 + (frequency / 700.0));

    let hertz (mel) =
        700.0 * exp ((mel / 1125.0) - 1.0)

    let mutable _filterbank : seq<triangle_filter> = null

    //See http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank
    let compute_filterbank (num_filters, fft_size, sample_rate) =
        if not (isNull _filterbank) then
            _filterbank
        else
            let lower_freq = 300.0;
            let upper_freq = 8000.0;
            let lower_mel = mel(lower_freq);
            let upper_mel = mel(upper_freq);
            let mel_spacing = (upper_mel - lower_mel) / float (num_filters + 1);
            let points_hertz = seq { for i in 0 .. (num_filters + 1) do yield hertz(lower_mel + (float i * mel_spacing)) };
            let fft_bin_points = points_hertz |> Seq.map(fun hertz_point -> int (floor ((float fft_size + 1.0) * float (hertz_point / sample_rate))));

            _filterbank <- seq {
                for i in 1 .. num_filters do
                    yield {
                        start_bin = fft_bin_points |> Seq.item(i - 1);
                        peak_bin = fft_bin_points |> Seq.item(i);
                        end_bin = fft_bin_points |> Seq.item(i + 1);
                    };
            };
            _filterbank

    //A more efficient Seq.sum(apply_filter())
    let apply_and_sum_filter (fft_output : float[], filter : triangle_filter) =
        let mutable total = 0.0;
        for i = filter.start_bin to filter.end_bin do
            if (i = filter.peak_bin) then
                total <- total + fft_output.[i];
            elif (i > filter.start_bin && i < filter.peak_bin) then
                let filter_weight = float (i - filter.start_bin) / float (filter.peak_bin - filter.start_bin);
                total <- total + fft_output.[i] * filter_weight;
            elif (i > filter.peak_bin && i < filter.end_bin) then
                let filter_weight = float (filter.end_bin - i) / float (filter.end_bin - filter.peak_bin);
                total <- total + fft_output.[i] * filter_weight;
        total;

    //https://en.wikipedia.org/wiki/Discrete_cosine_transform
    let dct (input:float[]) =
        let N = Array.length input;
        let output = Array.zeroCreate<float>(N);
        for k in 0 .. N - 1 do
            for n in 0 .. N - 1 do
                let xn = input.[n];
                Array.set output k (output.[k] + (xn * cos((float System.Math.PI / float N) * (float n + 0.5) * (float k))));

        Array.set output 0 (output.[0] * (1.0 / sqrt 2.0))
        let scale = sqrt (2.0 / float N)
        output |> Array.map(fun el -> el * scale)

    let compute (samples : float[], num_filters, num_features) =
        let hamming = Window.Hamming(Array.length samples);
        let windowed_samples = samples |> Array.mapi(fun index sample -> sample * hamming.[index]);

        let mutable complex_output = windowed_samples |> Array.map(fun sample -> Complex(sample, 0.0));
        Fourier.Forward(complex_output);
        let abs_output = complex_output |> Array.map(fun sample -> sqrt ((sample.Real * sample.Real) + (sample.Imaginary * sample.Imaginary)));
        //for extra optimization
        //let abs_output = complex_output |> Array.map(fun sample -> abs sample.Real + abs sample.Imaginary);

        let filters = compute_filterbank(num_filters, Array.length abs_output, 48000.0);

        let mel_output = filters |>
                            Seq.map(fun filter -> log10(apply_and_sum_filter(abs_output, filter))) |>
                            Seq.toArray;

        dct(mel_output) |> Array.take(num_features);