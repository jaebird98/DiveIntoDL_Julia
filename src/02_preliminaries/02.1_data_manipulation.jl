### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 38196006-5c99-11ec-0e33-a353d7719ec2
md"""
# 2.1. Data Manipulation

I will touch upon this subchapter very briefly.
The reader should know about Julia's column-first indexing and how functions work.
This part doesn't have anything interesting. moving on.
"""

# ╔═╡ edd736e6-dc3b-43d8-88d9-0e247ff45ede
md"""
## 2.1.1. Getting Started
Flux does not use tensors for automatic differentiation(AD). Instead, it uses source-to-source AD.
"""

# ╔═╡ b84cf546-fbf9-43a7-aa5f-f2c3e80794bf
# Returns an array of the given size filled with numbers sampled from normal distribution with mean of 0 and std of 1.
randn((3,4))

# ╔═╡ f5da1a29-0ae6-4987-a92b-6623b8baace0
# Note that Array function does not turn it from Vector of Vectors to a Matrix.
# For such operations, see cat, vcat, hcat, and reshape.
y = Array([[2.,1.,4.,3.],[1.,2.,3.,4.],[4.,3.,2.,1.]])

# ╔═╡ c1c32023-79f5-4b6f-a879-13baf1586d39
# The ... syntax means splatting, which means to feed each element inside a collection individually. 
Y = transpose(hcat(y...))

# ╔═╡ 577d9a6c-4e7a-4bc3-802b-4df159d4f8b1


# ╔═╡ Cell order:
# ╠═38196006-5c99-11ec-0e33-a353d7719ec2
# ╠═edd736e6-dc3b-43d8-88d9-0e247ff45ede
# ╠═b84cf546-fbf9-43a7-aa5f-f2c3e80794bf
# ╠═f5da1a29-0ae6-4987-a92b-6623b8baace0
# ╠═c1c32023-79f5-4b6f-a879-13baf1586d39
# ╠═577d9a6c-4e7a-4bc3-802b-4df159d4f8b1
