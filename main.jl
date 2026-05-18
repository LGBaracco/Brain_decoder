using NPZ, Lux, Random, Zygote, LuxCUDA, Optimisers, MLUtils

const device = gpu_device()


fmri_embs = npzread("data/train_data/subj01/training_split/training_fmri/lh_training_fmri.npy") |> device
# fmri_rh = npzread("data/train_data/subj01/training-split/lh_training_fmri.npy") let's do one hemisphere for now

clip_embs = npzread("data/clip_embeddings/train_vitl14.npy") |> device

encoder = Chain(
    Dense(19004, 512, relu),
    Dense(512, 256, relu),
    Dense(256, 128)         
)

ps, st = Lux.setup(Random.default_rng(), encoder) .|> device

# Custom InfoNCE loss definition
function infonce_loss(fmri_emb, clip_emb, τ=0.07)
    
    z_f = fmri_emb ./ (norm.(eachcol(fmri_emb))' .+ 1e-8)
    z_c = clip_emb ./ (norm.(eachcol(clip_emb))' .+ 1e-8)
    
    logits = (z_f' * z_c) ./ τ
    
    labels = 1:size(logits, 1)
    loss_i = mean(crossentropy(softmax(logits, dims=2), labels))
    loss_t = mean(crossentropy(softmax(logits, dims=1), labels))
    return (loss_i + loss_t) / 2
end

loader = DataLoader((fmri_embs', clip_embs'), batchsize=256, shuffle=true)

opt = Optimisers.Adam(3e-4)
opt_state = Optimisers.setup(opt, ps)

for (b, e) in loader
    # training step
    loss, grads = Zygote.withgradient(ps) do p
        z_fmri, _ = encoder(fmri_batch, ps, st)
        infonce_loss(z_fmri, clip_batch)
    end
    opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
end

