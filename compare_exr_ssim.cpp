#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define TINYEXR_IMPLEMENTATION
#include "MetalCpp Path Tracer/tinyexr.h"

namespace {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;
};

struct Options {
    bool include_alpha = false;
    bool luminance = false;
    int window_size = 11;
    double sigma = 1.5;
};

struct SSIMResult {
    double overall = 0.0;
    std::vector<double> per_channel;
};

std::vector<double> make_gaussian_kernel(int size, double sigma) {
    if (size % 2 == 0 || size <= 0) {
        throw std::invalid_argument("Gaussian kernel size must be positive and odd");
    }

    const int half = size / 2;
    std::vector<double> kernel(size * size);
    const double sigma2 = 2.0 * sigma * sigma;
    double sum = 0.0;
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            const double value = std::exp(-(x * x + y * y) / sigma2);
            kernel[(y + half) * size + (x + half)] = value;
            sum += value;
        }
    }
    if (sum <= 0.0) {
        throw std::runtime_error("Invalid Gaussian kernel normalization");
    }
    for (double &value : kernel) {
        value /= sum;
    }
    return kernel;
}

std::vector<double> convolve(const std::vector<double> &image, int width, int height,
                             const std::vector<double> &kernel, int kernel_size) {
    std::vector<double> output(width * height, 0.0);
    const int half = kernel_size / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int ky = 0; ky < kernel_size; ++ky) {
                const int yy = std::clamp(y + ky - half, 0, height - 1);
                const int kernel_row = ky * kernel_size;
                const int image_row = yy * width;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int xx = std::clamp(x + kx - half, 0, width - 1);
                    sum += image[image_row + xx] * kernel[kernel_row + kx];
                }
            }
            output[y * width + x] = sum;
        }
    }

    return output;
}

Image load_exr(const std::string &path, bool include_alpha) {
    float *rgba = nullptr;
    int width = 0;
    int height = 0;
    const char *err = nullptr;

    const int ret = LoadEXR(&rgba, &width, &height, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        std::string message = "Failed to load EXR: " + path;
        if (err) {
            message += " (";
            message += err;
            message += ")";
            FreeEXRErrorMessage(err);
        }
        throw std::runtime_error(message);
    }

    Image image;
    image.width = width;
    image.height = height;
    image.channels = include_alpha ? 4 : 3;
    image.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * image.channels);

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int channel = 0; channel < image.channels; ++channel) {
            image.data[i * image.channels + channel] = rgba[i * 4 + channel];
        }
    }

    free(rgba);
    return image;
}

Image to_luminance(const Image &image) {
    if (image.channels < 3) {
        throw std::runtime_error("Luminance conversion requires at least three channels");
    }

    Image luminance;
    luminance.width = image.width;
    luminance.height = image.height;
    luminance.channels = 1;
    luminance.data.resize(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));

    for (size_t i = 0; i < luminance.data.size(); ++i) {
        const float r = image.data[i * image.channels + 0];
        const float g = image.data[i * image.channels + 1];
        const float b = image.data[i * image.channels + 2];
        luminance.data[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    return luminance;
}

SSIMResult compute_ssim(const Image &a, const Image &b, const std::vector<double> &kernel,
                        int kernel_size) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) {
        throw std::runtime_error("Images must have the same dimensions and channel count");
    }

    const int width = a.width;
    const int height = a.height;
    const int channels = a.channels;
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

    SSIMResult result;
    result.per_channel.reserve(channels);

    for (int channel = 0; channel < channels; ++channel) {
        std::vector<double> channel_a(pixel_count);
        std::vector<double> channel_b(pixel_count);
        double min_value = std::numeric_limits<double>::infinity();
        double max_value = -std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < pixel_count; ++i) {
            const double value_a = static_cast<double>(a.data[i * channels + channel]);
            const double value_b = static_cast<double>(b.data[i * channels + channel]);
            channel_a[i] = value_a;
            channel_b[i] = value_b;
            min_value = std::min({min_value, value_a, value_b});
            max_value = std::max({max_value, value_a, value_b});
        }

        double dynamic_range = max_value - min_value;
        if (dynamic_range < 1e-6) {
            dynamic_range = 1.0;
        }
        const double c1 = std::pow(0.01 * dynamic_range, 2.0);
        const double c2 = std::pow(0.03 * dynamic_range, 2.0);

        std::vector<double> mu_a = convolve(channel_a, width, height, kernel, kernel_size);
        std::vector<double> mu_b = convolve(channel_b, width, height, kernel, kernel_size);

        std::vector<double> mu_a_sq(mu_a.size());
        std::vector<double> mu_b_sq(mu_b.size());
        std::vector<double> mu_a_mu_b(mu_a.size());
        for (size_t i = 0; i < mu_a.size(); ++i) {
            mu_a_sq[i] = mu_a[i] * mu_a[i];
            mu_b_sq[i] = mu_b[i] * mu_b[i];
            mu_a_mu_b[i] = mu_a[i] * mu_b[i];
        }

        std::vector<double> channel_a_sq(pixel_count);
        std::vector<double> channel_b_sq(pixel_count);
        std::vector<double> channel_ab(pixel_count);
        for (size_t i = 0; i < pixel_count; ++i) {
            channel_a_sq[i] = channel_a[i] * channel_a[i];
            channel_b_sq[i] = channel_b[i] * channel_b[i];
            channel_ab[i] = channel_a[i] * channel_b[i];
        }

        std::vector<double> sigma_a_sq = convolve(channel_a_sq, width, height, kernel, kernel_size);
        std::vector<double> sigma_b_sq = convolve(channel_b_sq, width, height, kernel, kernel_size);
        std::vector<double> sigma_ab = convolve(channel_ab, width, height, kernel, kernel_size);

        double channel_ssim_sum = 0.0;
        for (size_t i = 0; i < mu_a.size(); ++i) {
            sigma_a_sq[i] = std::max(0.0, sigma_a_sq[i] - mu_a_sq[i]);
            sigma_b_sq[i] = std::max(0.0, sigma_b_sq[i] - mu_b_sq[i]);
            sigma_ab[i] -= mu_a_mu_b[i];

            const double numerator = (2.0 * mu_a_mu_b[i] + c1) * (2.0 * sigma_ab[i] + c2);
            const double denominator = (mu_a_sq[i] + mu_b_sq[i] + c1) *
                                       (sigma_a_sq[i] + sigma_b_sq[i] + c2);
            double ssim = 1.0;
            if (denominator > 1e-12) {
                ssim = numerator / denominator;
            }
            channel_ssim_sum += ssim;
        }

        const double channel_ssim = channel_ssim_sum / static_cast<double>(mu_a.size());
        result.per_channel.push_back(channel_ssim);
    }

    double sum = 0.0;
    for (double value : result.per_channel) {
        sum += value;
    }
    result.overall = sum / static_cast<double>(result.per_channel.size());

    return result;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 3; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--include-alpha") {
            options.include_alpha = true;
        } else if (arg == "--luminance") {
            options.luminance = true;
        } else if (arg == "--window-size") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --window-size");
            }
            options.window_size = std::stoi(argv[++i]);
        } else if (arg == "--sigma") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --sigma");
            }
            options.sigma = std::stod(argv[++i]);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.window_size <= 0 || options.window_size % 2 == 0) {
        throw std::runtime_error("Window size must be a positive odd integer");
    }
    if (options.sigma <= 0.0) {
        throw std::runtime_error("Sigma must be positive");
    }
    return options;
}

void print_usage(const char *program) {
    std::cerr << "Usage: " << program << " <reference.exr> <test.exr> [options]\n"
              << "Options:\n"
              << "  --include-alpha   Include alpha channel in SSIM computation (default: ignore)\n"
              << "  --luminance       Convert RGB to luminance before computing SSIM\n"
              << "  --window-size N   Gaussian kernel size (odd integer, default: 11)\n"
              << "  --sigma S         Gaussian kernel sigma (default: 1.5)\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const std::string reference_path(argv[1]);
        const std::string test_path(argv[2]);
        const Options options = parse_options(argc, argv);

        Image reference = load_exr(reference_path, options.include_alpha);
        Image test = load_exr(test_path, options.include_alpha);

        if (options.luminance) {
            reference = to_luminance(reference);
            test = to_luminance(test);
        }

        const std::vector<double> kernel = make_gaussian_kernel(options.window_size, options.sigma);
        const SSIMResult result = compute_ssim(reference, test, kernel, options.window_size);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Overall SSIM: " << result.overall << "\n";
        for (size_t i = 0; i < result.per_channel.size(); ++i) {
            std::cout << "Channel " << i << " SSIM: " << result.per_channel[i] << "\n";
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}

