
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux, CUDA, MAT, FFTW
using Flux.Losses
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: norm
using MLDataPattern: splitobs


if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
    device!(2)
end

## ---
# Structs
@with_kw mutable struct Args
    batchsize::Int = 50
    valsplit::Float64 = 0.1
    patience::Int = 1
    target_error::Float64 = 1e-1

    lr::Float64 = 2e-6
    moment::Float64 = 0.6
    dropout::Float64 = 0.004

    patch_size::Int = 64
    input_shape = (patch_size, patch_size, 2)
    output_shape = (patch_size, patch_size)

    File_path_train = "./data/train_input.mat"
    Variable_name_train = "train_fft_x"

    File_path_test = "./data/test_input.mat"
    Variable_name_test = "test_fft_x"
end

## ---
# Helper Functions
function Load_mat_images(File_path,Variable_name)
    file = matopen(File_path)
    x_in = read(file,Variable_name)
    close(file)
    return x_in
end
function process_data(args,File_path,Variable_name)
    # Used to calculate the FFT
    x_in = Load_mat_images(File_path,Variable_name)
    x_in_tmp = reshape(x_in,(args.patch_size,args.patch_size,2,size(x_in,2)))

    y_in_tmp = x_in_tmp[:,:,1,:] + 1im.*(x_in_tmp[:,:,2,:])
    y_in = Array{Float32}(undef,size(y_in_tmp,1),size(y_in_tmp,2),size(y_in_tmp,3))
    for index in 1:size(y_in_tmp,3)
        y_in[:,:,index] = abs.(ifftshift(fft(y_in_tmp[:,:,index])))
    end
    return Float32.(x_in),Float32.(y_in)
end
function get_data_train(args,File_path_train,Variable_name_train)
    x_in, y_in = process_data(args,File_path_train,Variable_name_train)
    y_in = reshape(y_in,(size(y_in,1)*size(y_in,2),size(y_in,3)))

    @show size(x_in)
    @show size(y_in)
    (train_x, train_y), (val_x, val_y) = splitobs((x_in,y_in), at=1-args.valsplit)

    return (train_x, train_y), (val_x, val_y)
end

## ---
# Model
function automap(Args)
    Chain(  
        Dense((Args.patch_size).^2*2, (Args.patch_size).^2, tanh),
        Dropout(Args.dropout),
        Dense((Args.patch_size).^2, (Args.patch_size).^2, tanh),
        Dropout(Args.dropout),
        x -> reshape(x, (Args.patch_size,Args.patch_size,1,:)),
        Conv((5,5), 1 => Args.patch_size, relu; stride = 1, pad = 2),
        Dropout(Args.dropout),
        Conv((5,5), Args.patch_size => Args.patch_size, relu; stride = 1, pad = 2),
        Dropout(Args.dropout),
        ConvTranspose((7,7), Args.patch_size => 1; stride = 1, pad = 3),
        flatten,
    )
end
function train_while(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)

    # Load the train, validation data 
    train_data, val_data = get_data_train(args,args.File_path_train,args.Variable_name_train)
    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")	
    m = (automap(args)) |> gpu
    
    # Loss Function
    loss(x, y) = mse(m(x), y)

    # Training
    # Defining the optimizer
    opt = RMSProp(args.lr,args.moment)
    ps = Flux.params(m)

    epoch, mse_val_lowest, epoch_with_lowest_val_mse = 0,0,0
    is_training = true

    @info("Training....")
    # Starting to train models
    while is_training == true
        epoch = epoch + 1
        batch_counter = 0
        @info "Epoch $epoch"

        for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(() -> loss(x,y), ps)
            Flux.update!(opt, ps, gs)
        end

        validation_loss = 0f0
        for (x, y) in val_loader
            x, y = x |> gpu, y |> gpu
            validation_loss += loss(x, y)
        end
        validation_loss /= length(val_loader)
        @show validation_loss

        # Early Stopping
        if abs.(mse_val_lowest-validation_loss) < args.target_error
            # no improvement
            @info "Delta Error is < $(args.target_error) since $(epoch - epoch_with_lowest_val_mse) of $(args.patience) epochs"
            @info "Delta Error value: $(abs.(mse_val_lowest-validation_loss))"
            if epoch >= epoch_with_lowest_val_mse + args.patience
                # stop training
                is_training = false
            end
        else
            # improvement
            epoch_with_lowest_val_mse = epoch
            mse_val_lowest = validation_loss
        end
    end
    return m
end
function Run_save_model()
    m = @time train_while()
    return cpu(m)
end
## ---
#Training the Model
model = Run_save_model()

## ---

args = Args()

x_in, y_in = process_data(args,args.File_path_train,args.Variable_name_train)
@show (size(x_in),size(y_in))

y_in = reshape(y_in,(size(y_in,1)*size(y_in,2),size(y_in,3)))

# Load the train, validation data 
train_data, val_data = get_data_train(args,args.File_path_train,args.Variable_name_train)
train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)

for (x, y) in train_loader
    @show size(x)
    @show size(y)
end

val_loader = DataLoader(val_data, batchsize=args.batchsize)