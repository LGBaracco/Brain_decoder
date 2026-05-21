using NPZ, Lux, Random, Zygote, LuxCUDA, Optimisers, MLUtils, LinearAlgebra, NNlib, Statistics, StatsBase

# TODO fix, maybe switch to flux or even pytorch, too many libs
# Status: the code runs, but at what cost? NaN

# Custom InfoNCE loss definition
function infonce_loss(fmri_emb, clip_emb, τ=0.07f0)
    
    z_f = fmri_emb ./ (sqrt.(sum(abs2, fmri_emb, dims=1)) .+ 1e-8)
    z_c = clip_emb ./ (sqrt.(sum(abs2, clip_emb, dims=1)) .+ 1e-8)
    
    logits = (z_f' * z_c) ./ τ
    
    #labels = 1:size(logits, 1)
    n = size(logits, 1)
    labels = Zygote.ignore() do
        device(Matrix{Float32}(I, n, n))
    end

    loss_i = mean(crossentropy(softmax(logits, dims=2), labels))
    loss_t = mean(crossentropy(softmax(logits, dims=1), labels))


    return (loss_i + loss_t) / 2
end

function train!(encoder, fmri_embs, clip_embs; n_epochs=50, batchsize=256)
    ps, st = Lux.setup(Random.default_rng(), encoder) .|> device


    loader = DataLoader((fmri_embs', clip_embs'), batchsize=batchsize, shuffle=true)

    opt = Optimisers.Adam(3e-4)
    opt_state = Optimisers.setup(opt, ps)

    for epoch in 1:n_epochs
        epoch_loss = 0f0
        n_batches  = 0

        for (fmri_batch, clip_batch) in loader
            fmri_batch = device(fmri_batch)
            clip_batch = device(clip_batch)

            loss, grads = Zygote.withgradient(ps) do p
                z_fmri, _ = encoder(fmri_batch, p, st)
                infonce_loss(z_fmri, clip_batch)
            end
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

            n_batches  += 1
            epoch_loss += loss
        end
        @info "Epoch $epoch" loss=epoch_loss/n_batches
    end
    return ps, st
end

const device = gpu_device()

fmri_embs = npzread("data/train_data/subj01/training_split/training_fmri/lh_training_fmri.npy") |> device
# fmri_rh = npzread("data/train_data/subj01/training-split/lh_training_fmri.npy") let's do one hemisphere for now

clip_embs = npzread("data/clip_embeddings/train_vitl14.npy") |> device

encoder = Chain(
    Dense(19004, 512, relu),
    Dense(512, 256, relu),
    Dense(256, 768)         
)

trained_ps, trained_st = train!(encoder, fmri_embs, clip_embs)