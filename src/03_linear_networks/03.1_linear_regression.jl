### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ afcc3e0e-2383-4cd6-8d8e-7ac330686e4f
using BenchmarkTools

# ╔═╡ 159ad7e6-5be3-11ec-390a-95ef10649fa8
md"""
# 3.1 Linear Regression

"""

# ╔═╡ f6004a90-5f07-45d2-94be-1dce49a994da
md"""
## 3.1.2. Vectorization for Speed
Nevertheless, in order to see how to benchmark operations, we will go ahead with this part anyway.

Base julia has simple benchmark tools, but a better option is package BenchmarkTools.
If you followed README.md, then you should have it installed in the environment.
"""

# ╔═╡ 78f593be-57d6-40fa-9dcf-2f6309353460
md"""
Now we will define a function that adds two arrays together in a for loop, and time it.
"""

# ╔═╡ 56d650d6-2f0c-462e-8337-abb640c51980
begin
	n = 10000
	a = ones(n)
	b = ones(n)
	c = zeros(n)
end

# ╔═╡ f5f6aee8-3f34-4bfb-9cab-880b2773255a
@benchmark begin
	for i in eachindex(a)
		c[i] = a[i] + b[i]
	end
end

# ╔═╡ 41012057-568a-47a4-bef8-364dea9d79e3
md"""
Now we will use broadcasting operator, and see how julia compiler fares.
"""

# ╔═╡ 80bd2ccd-ec10-4343-9dd8-ca2d475f2059
@benchmark begin
	d = a .+ b
end

# ╔═╡ 82e1be7f-0824-4537-a22b-2b1247041ec7
md"""
I'm sure I am doing something wrong, as in julia for loops should be efficiently compiled. This probably has to do with the fact that c is explicitly allocated in for loops, while d is not, so allocation doesn't happen.
"""

# ╔═╡ 84a31c4c-8f4f-4b81-a4b4-1b149cffff36


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"

[compat]
BenchmarkTools = "~1.2.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╠═159ad7e6-5be3-11ec-390a-95ef10649fa8
# ╠═f6004a90-5f07-45d2-94be-1dce49a994da
# ╠═afcc3e0e-2383-4cd6-8d8e-7ac330686e4f
# ╠═78f593be-57d6-40fa-9dcf-2f6309353460
# ╠═56d650d6-2f0c-462e-8337-abb640c51980
# ╠═f5f6aee8-3f34-4bfb-9cab-880b2773255a
# ╠═41012057-568a-47a4-bef8-364dea9d79e3
# ╠═80bd2ccd-ec10-4343-9dd8-ca2d475f2059
# ╠═82e1be7f-0824-4537-a22b-2b1247041ec7
# ╠═84a31c4c-8f4f-4b81-a4b4-1b149cffff36
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
