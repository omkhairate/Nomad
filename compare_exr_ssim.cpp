#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define TINYEXR_IMPLEMENTATION
#include "MetalCpp Path Tracer/tinyexr.h"

namespace {

namespace fs = std::filesystem;

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
    fs::path excel_output = "ssim_scores.csv";
};

struct ReportRow {
    std::string image_name;
    double ssim = 0.0;
};

struct SSIMResult {
    double overall = 0.0;
    std::vector<double> per_channel;
};

bool has_exr_extension(const fs::path &path) {
    const std::string extension = path.extension().string();
    if (extension.empty()) {
        return false;
    }

    std::string lower;
    lower.reserve(extension.size());
    for (char ch : extension) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return lower == ".exr";
}

std::vector<std::pair<fs::path, fs::path>> match_exr_files(const fs::path &reference_dir,
                                                           const fs::path &test_dir) {
    std::map<std::string, fs::path> reference_files;
    std::map<std::string, fs::path> test_files;

    for (const auto &entry : fs::directory_iterator(reference_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path &path = entry.path();
        if (!has_exr_extension(path)) {
            continue;
        }

        const std::string filename = path.filename().string();
        const auto insert_result = reference_files.emplace(filename, path);
        if (!insert_result.second) {
            throw std::runtime_error("Duplicate reference EXR filename: " + filename);
        }
    }

    for (const auto &entry : fs::directory_iterator(test_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path &path = entry.path();
        if (!has_exr_extension(path)) {
            continue;
        }

        const std::string filename = path.filename().string();
        const auto insert_result = test_files.emplace(filename, path);
        if (!insert_result.second) {
            throw std::runtime_error("Duplicate test EXR filename: " + filename);
        }
    }

    std::vector<std::pair<fs::path, fs::path>> pairs;
    pairs.reserve(reference_files.size());

    for (const auto &[filename, reference_path] : reference_files) {
        const auto it = test_files.find(filename);
        if (it == test_files.end()) {
            std::cerr << "Warning: Missing test EXR for '" << filename << "'\n";
            continue;
        }

        pairs.emplace_back(reference_path, it->second);
    }

    for (const auto &[filename, test_path] : test_files) {
        if (reference_files.find(filename) == reference_files.end()) {
            std::cerr << "Warning: Missing reference EXR for '" << filename << "'\n";
        }
    }

    return pairs;
}

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
        } else if (arg == "--excel-out") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --excel-out");
            }
            options.excel_output = argv[++i];
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
    std::cerr << "Usage: " << program << " <reference> <test> [options]\n"
              << "Compare EXR files directly or directories containing EXR files.\n"
              << "Options:\n"
              << "  --include-alpha   Include alpha channel in SSIM computation (default: ignore)\n"
              << "  --luminance       Convert RGB to luminance before computing SSIM\n"
              << "  --window-size N   Gaussian kernel size (odd integer, default: 11)\n"
              << "  --sigma S         Gaussian kernel sigma (default: 1.5)\n"
              << "  --excel-out PATH  Write Excel-compatible CSV with per-file SSIM (default: ssim_scores.csv)\n";
}

std::string escape_csv_field(const std::string &value) {
    bool needs_quotes = false;
    for (char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return value;
    }

    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += '"';
        }
        escaped += ch;
    }
    escaped += '"';
    return escaped;
}

void write_csv_report(const fs::path &output_path, const std::vector<ReportRow> &rows) {
    if (rows.empty()) {
        return;
    }

    std::ofstream file(output_path);
    if (!file) {
        throw std::runtime_error("Failed to open Excel output file: " + output_path.string());
    }

    file << "Image Name,SSIM\n";
    file << std::fixed << std::setprecision(6);
    for (const auto &row : rows) {
        file << escape_csv_field(row.image_name) << ',' << row.ssim << "\n";
    }

    if (!file) {
        throw std::runtime_error("Failed while writing Excel output file: " + output_path.string());
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const fs::path reference_path(argv[1]);
        const fs::path test_path(argv[2]);
        const Options options = parse_options(argc, argv);

        const std::vector<double> kernel = make_gaussian_kernel(options.window_size, options.sigma);
        if (!fs::exists(reference_path)) {
            throw std::runtime_error("Reference path does not exist: " + reference_path.string());
        }
        if (!fs::exists(test_path)) {
            throw std::runtime_error("Test path does not exist: " + test_path.string());
        }

        std::vector<ReportRow> report_rows;

        if (fs::is_regular_file(reference_path) && fs::is_regular_file(test_path)) {
            Image reference = load_exr(reference_path.string(), options.include_alpha);
            Image test = load_exr(test_path.string(), options.include_alpha);

            if (options.luminance) {
                reference = to_luminance(reference);
                test = to_luminance(test);
            }

            const SSIMResult result = compute_ssim(reference, test, kernel, options.window_size);

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Overall SSIM: " << result.overall << "\n";
            for (size_t i = 0; i < result.per_channel.size(); ++i) {
                std::cout << "Channel " << i << " SSIM: " << result.per_channel[i] << "\n";
            }

            report_rows.push_back({reference_path.filename().string(), result.overall});
            write_csv_report(options.excel_output, report_rows);
            std::cout << "Wrote Excel report to: " << options.excel_output << "\n";
            return EXIT_SUCCESS;
        }

        if (fs::is_directory(reference_path) && fs::is_directory(test_path)) {
            const auto pairs = match_exr_files(reference_path, test_path);
            if (pairs.empty()) {
                throw std::runtime_error("No matching EXR files found between directories");
            }

            std::cout << std::fixed << std::setprecision(6);
            double overall_sum = 0.0;
            size_t compared = 0;

            for (const auto &[ref_file, test_file] : pairs) {
                Image reference = load_exr(ref_file.string(), options.include_alpha);
                Image test = load_exr(test_file.string(), options.include_alpha);

                if (options.luminance) {
                    reference = to_luminance(reference);
                    test = to_luminance(test);
                }

                const SSIMResult result = compute_ssim(reference, test, kernel, options.window_size);

                std::cout << "File: " << ref_file.filename().string() << "\n";
                std::cout << "  Overall SSIM: " << result.overall << "\n";
                for (size_t i = 0; i < result.per_channel.size(); ++i) {
                    std::cout << "  Channel " << i << " SSIM: " << result.per_channel[i] << "\n";
                }
                std::cout << '\n';

                overall_sum += result.overall;
                ++compared;

                report_rows.push_back({ref_file.filename().string(), result.overall});
            }

            if (compared > 1) {
                const double average = overall_sum / static_cast<double>(compared);
                std::cout << "Average overall SSIM: " << average << "\n";
            }

            write_csv_report(options.excel_output, report_rows);
            std::cout << "Wrote Excel report to: " << options.excel_output << "\n";
            return EXIT_SUCCESS;
        }

        throw std::runtime_error("Both paths must be files or both must be directories");
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}

