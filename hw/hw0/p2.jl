println("booting julia...")
using ClusterManagers, Distributed
using DataFrames, CSV
println("done.")

println("demanding workers from slurm...")
addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))
println("done. added workers: ", nworkers())

println("prepping computation...")
@everywhere begin
    using Random

    # brownian motion
    make_brownian(dt,n) = cumsum([0;sqrt(dt).*randn(n+1)])

    # get the walk for some seed, and grab the hostname as well
    get_results(seed) = begin
        Random.seed!(seed)
        brownian = make_brownian(1, 100)
        hostname = gethostname()
        (brownian, hostname)
    end
end
println("computation prepared.")

println("running...")
seeds = 1:50
results = pmap(get_results, seeds)
println("done.")

println("writing output...")
function format_results(results :: Array{Tuple{Array{Float64, 1}, String}, 1})
    all = DataFrame(time = [], y = [], hostname = [])
    for (walk, hostname) in results
        current = DataFrame(time = 1:length(walk), y = walk, hostname = [hostname for i in 1:length(walk)])
        append!(all, current)
    end
    all
end

CSV.write("results.csv", format_results(results))
println("done.")
