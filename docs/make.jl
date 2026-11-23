# Build script for the GaussianEP.jl documentation.
#
# Run from the repository root with the docs environment active:
#
#     julia --project=docs docs/make.jl
#
# Output is written to docs/build/ (open docs/build/index.html in a browser).

using Documenter, Literate, GaussianEP

const DOCS_DIR = @__DIR__
const NOTEBOOKS_IN = joinpath(DOCS_DIR, "notebooks")         # docs/notebooks/*.jl
const NOTEBOOKS_OUT = joinpath(DOCS_DIR, "src", "notebooks")  # docs/src/notebooks/*.md

# Render notebooks with Literate: execute docs/notebooks/*.jl and write
# Documenter-flavoured markdown to docs/src/notebooks/*.md.
# Guarded with isdir so the build works before any notebook has landed.
notebook_pages = String[]
if isdir(NOTEBOOKS_IN)
    mkpath(NOTEBOOKS_OUT)
    # drop stale generated pages (e.g. from a removed/renamed notebook)
    for stale in readdir(NOTEBOOKS_OUT; join = true)
        endswith(stale, ".md") && rm(stale; force = true)
    end
    for file in sort(readdir(NOTEBOOKS_IN; join = true))
        if endswith(file, ".jl")
            out = Literate.markdown(file, NOTEBOOKS_OUT; documenter = true, execute = true)
            push!(notebook_pages, "notebooks/" * basename(out))
        end
    end
end

pages = Any[
    "Home" => "index.md",
    "API" => "api.md",
    "Notebooks" => "notebooks.md",
]
for p in notebook_pages
    title = basename(p) |> f -> replace(f, r"\.md$" => "") |> f -> replace(f, "_" => " ")
    push!(pages, title => p)
end

makedocs(
    sitename = "GaussianEP.jl",
    modules = [GaussianEP],
    doctest = false,
    format = Documenter.HTML(),
    pages = pages,
    # internal helpers with docstrings that are not (yet) in the manual
    warnonly = [:missing_docs],
)
