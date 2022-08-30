begin
    ims = []
    for (root, dirs, files) in walkdir(datapath)
        for (i, file) in enumerate(files)
            if endswith(file, ".png") && !(occursin("map", file))
                img = load(joinpath(root, file))
                img = imresize(img, (50, 50))
                push!(ims, permutedims(Float32.(channelview(img)), [2, 3, 1]))
            end
        end
    end
end