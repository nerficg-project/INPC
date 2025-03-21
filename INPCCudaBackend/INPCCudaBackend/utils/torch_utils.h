#pragma once

#include <functional>
#include <torch/extension.h>

std::function<char*(size_t N)> resize_function_wrapper(torch::Tensor& t);
