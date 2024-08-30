#include "op/layer.h"
#include <cstdarg>
#include "stdio.h"

op::BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, \
    base::DataType data_type, std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name))
{
}

base::DataType op::BaseLayer::data_type() const
{
    return data_type_;
}

op::LayerType op::BaseLayer::layer_type() const
{
    return layer_type_;
}

base::Status op::BaseLayer::set_weight(int32_t idx, const tensor::Tensor &weight)
{
    return base::error::FunctionNotImplement();
}

base::Status op::BaseLayer::set_weight(int32_t idx, const std::vector<int32_t> &dims, \
    const void *weight_ptr, base::DeviceType device_type)
{
    return base::error::FunctionNotImplement();
}

const std::string &op::BaseLayer::get_layer_name() const
{
    return layer_name_;
}

void op::BaseLayer::set_layer_name(const std::string &layer_name)
{
    layer_name_ = layer_name;
}

base::DeviceType op::BaseLayer::device_type() const
{
    return device_type_;
}

void op::BaseLayer::set_device_type(base::DeviceType device_type)
{
    device_type_ = device_type;
}

op::LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, \
        bool is_quant_layer, std::string layer_name)
        : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer)
{

}

size_t op::LayerParam::weight_size() const
{
    return weights_.size();
}

void op::LayerParam::reset_weight_size(size_t size)
{
    weights_.resize(size);
}

tensor::Tensor &op::LayerParam::get_weight(int32_t idx)
{
    CHECK_GE(idx,0);
    CHECK_LT(idx,weights_.size());
    return weights_.at(idx);
}

const tensor::Tensor &op::LayerParam::get_weight(int32_t idx) const
{
    CHECK_GE(idx,0);
    CHECK_LT(idx,weights_.size());
    return weights_.at(idx);
}

base::Status op::LayerParam::set_weight(int32_t idx, const tensor::Tensor &weight)
{
    return base::error::FunctionNotImplement();
}

base::Status op::LayerParam::set_weight(int32_t idx, const std::vector<int32_t> &dims, const void *weight_ptr, base::DeviceType device_type)
{
    return base::error::FunctionNotImplement();
}
void op::LayerParam::set_scales(const tensor::Tensor &scales)
{
    CHECK(!scales.is_empty());
    this->scales_ = scales;
}

void op::LayerParam::set_group_size(int32_t group_size)
{
    this->group_size_ = group_size; 
}

int32_t op::LayerParam::get_scale_num() const
{
    CHECK(!scales_.is_empty());
    return static_cast<int32_t>(scales_.size());
}
/******************************** */
// op::Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
// {
//     ;
// }

base::Status op::Layer::init()
{
    return base::error::Success();
}

base::Status op::Layer::check_tensor(const tensor::Tensor &tensor, base::DeviceType device_type, base::DataType data_type) const
{
    if (tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }
    return base::error::Success();
}

base::Status op::Layer::check_tensor_with_dim(const tensor::Tensor &tensor, base::DeviceType device_type, base::DataType data_type, ...) const
{
    std::va_list args;
    if (tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }

    va_start(args, data_type);
    int32_t dims = tensor.dims_size();
    for (int32_t i = 0; i < dims; ++i) {
        int32_t dim = va_arg(args, int32_t);
        if (dim != tensor.get_dim(i)) {
            return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
        }
    }
    va_end(args);
    return base::error::Success();
}

base::Status op::Layer::check() const
{
    return base::error::FunctionNotImplement("The check function is not implement yet");
}

base::Status op::Layer::forward()
{
    return base::error::FunctionNotImplement("");
}

base::Status op::Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &output1)
{
    this->set_input(0,input1);
    this->set_output(0,output1);
    return this->forward();
}

base::Status op::Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2, \
                    const tensor::Tensor &output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);

    this->set_output(0, output1);
    return this->forward();
}

base::Status op::Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2, \
                    const tensor::Tensor &input3, const tensor::Tensor &output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);

    this->set_output(0, output1);
    return this->forward();
}

base::Status op::Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,\
             const tensor::Tensor &input3, const tensor::Tensor &input4, const tensor::Tensor &output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);

    this->set_output(0, output1);
    return this->forward();
}

base::Status op::Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2, \
            const tensor::Tensor &input3, const tensor::Tensor &input4, const tensor::Tensor &input5, const tensor::Tensor &output1)
{
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

void op::Layer::set_input(int32_t idx, const tensor::Tensor &input)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    this->inputs_.at(idx) = input;
}

void op::Layer::set_output(int32_t idx, const tensor::Tensor &output)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    this->outputs_.at(idx) = output;
}

const tensor::Tensor &op::Layer::get_input(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

const tensor::Tensor &op::Layer::get_output(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

tensor::Tensor &op::Layer::get_input(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

tensor::Tensor &op::Layer::get_output(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

size_t op::Layer::input_size() const
{
    return inputs_.size();
}

size_t op::Layer::output_size() const
{
    return outputs_.size(); 
}

void op::Layer::reset_input_size(size_t size)
{
    inputs_.resize(size);
}

void op::Layer::reset_output_size(size_t size)
{
    outputs_.resize(size); 
}
