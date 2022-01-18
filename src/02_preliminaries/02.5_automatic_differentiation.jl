### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 4a928227-430e-4709-9223-6357b67a27b3
begin
	using Zygote
	using LinearAlgebra
end

# ╔═╡ 8ce6ee24-5d79-11ec-27c4-23683f962d19
md"""
# 2.5. Automatic Differentiation

Since Derivatives of functions are so important in DL, one naturally seeks out how to automate it.

Here we use Zygote.jl, a source-to-source AD package, upon which Flux.jl is built.
"""

# ╔═╡ e571b4a7-6236-4abc-ba18-d5e03441dfd9
md"""
## 2.5.1. A simple example
Let us illustrate how Zygote is used.
"""

# ╔═╡ acc5a91e-cd4a-45d8-89d3-ea9a846d0492
x = [1, 2, 3, 4]

# ╔═╡ 7e4a6ab4-70c6-44c4-ad3b-311a07a74092
# f'(x) is 4x
f(x) = 2 * dot(x, x)

# ╔═╡ 8a0d5c93-46c0-4c9c-ab13-71a039239f8c
f(x)

# ╔═╡ 8078b080-a268-4ca5-ad7a-6adda8d193ab
# That is basically it. Load Zygote, and it will automatically differentiate for you.
f'(x) == 4x

# ╔═╡ 6d0bfcd0-6879-4ea1-ba32-f4a74cbda890
md"""
## 2.5.2 Backward for Non-Scalar Variables

In ML, differentiation of higher order tensors are tensors(jacobians, to be precise),
Zygote does provide jacobians, but usually what we want to do is to compute gradient of a vector, which in this context is a batch of training examples.

Let us see how to achieve that.
"""

# ╔═╡ d67db15c-b26e-4f20-a882-32a45fc74411
f1(x) = x * x

# ╔═╡ 5887063d-e936-4b6b-b589-48718b4a3f33
f1.(x)

# ╔═╡ 3fae0245-3823-41d6-99dc-e3ac7639fe6d
f1'.(x)

# ╔═╡ faddb3c9-f059-495e-85c8-64fbb60080ff
md"""
Unlike python, julia has an explicit syntax for broadcasting, as long as you define a function which returns a scalar and put a dot in it, Zygote will know that you are broadcasting and return values accordingly.
"""

# ╔═╡ 18d02e17-7864-4653-a40e-2b0ca47e8849
md"""
## 2.5.3. Detaching Computation

detaching computation means to take an intermediary value as a constant during backpropagation.

Zygote has a dropgrad function as an equivalent of detach function in pytorch.
However, it has been noted to be somewhat unstable.
Therefore, along with an example of dropgrad, we will also show how to detach computation manually.
"""

# ╔═╡ 59f33b66-8812-48d3-b50b-186bb033f442
# example of dropgrad function
begin
	z(x) = Zygote.dropgrad(x^2) * x
	z'.(x) == x.^2
end

# ╔═╡ b27d40b9-1f2c-4c7c-b852-e8d549653117
# example of manual detach
begin
	detach(f) = f()
	Zygote.@nograd detach

	f2(x) = x*x
	f3(x) = detach() do
		f2(x)
	end
	f4(x) = f3(x) * x
	f4'.(x) == f3.(x)
end

# ╔═╡ 14f41921-3dbe-4817-935b-0731c0531624
md"""
# 2.5.4. Computing Gradient of Control Flows

Here we don't show off any new ideas, but rather show that Zygote works on functions that have control flows.
"""

# ╔═╡ d0467970-7f6b-4fd4-98e5-5925398e32d9
function g(a)
	b = a * 2
	while norm(b) < 1000
		b = b * 2
	end

	if sum(b) > 0
		c = b
	else
		c = 100 * b
	end

	return c
end

# ╔═╡ d0137c19-2928-4233-a398-4555cc11e155
a = randn()

# ╔═╡ 4c877223-6efb-417d-9272-63cac89af785
g'(a) == g(a)/a

# ╔═╡ 3b29dc0e-e2c7-4694-9259-e3a82ed7670f
md"""
# 2.5.5. Summary

Deep learning frameworks can automate the calculation of derivatives. Flux.jl accomplishes this using source code transformation. It is quite abstract and powerful, but has some kinks to work out.
"""

# ╔═╡ ca3508b8-4fab-462f-ab55-9be8a041ebc4
md"""
# 2.5.6. Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?

2. After running the function for backpropagation, immediately run it again and see what happens.

3. In the control flow example where we calculate the derivatives of d with respect to a, what would happen if we changed the variable a to a random vector or matrix. At this point, the result of the calculation f(a) is no longer a scalar. What happens to the result? How do we analyze this?

4. Redesign an example of findin
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Zygote = "~0.6.32"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "RealDot", "Statistics"]
git-tree-sha1 = "feeac82d7ef2bc0e531433a1f1bd65b4d8dd53c8"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.16.0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "9bc5dac3c8b6706b58ad5ce24cffd9861f07c94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2b72a5624e289ee18256111657663721d59c143e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.24"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "006127162a51f0effbdfaab5ac0c83f8eb7ea8f3"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.4"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "76475a5aa0be302c689fd319cd257cd1a512fb3c"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.32"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═8ce6ee24-5d79-11ec-27c4-23683f962d19
# ╠═4a928227-430e-4709-9223-6357b67a27b3
# ╠═e571b4a7-6236-4abc-ba18-d5e03441dfd9
# ╠═acc5a91e-cd4a-45d8-89d3-ea9a846d0492
# ╠═7e4a6ab4-70c6-44c4-ad3b-311a07a74092
# ╠═8a0d5c93-46c0-4c9c-ab13-71a039239f8c
# ╠═8078b080-a268-4ca5-ad7a-6adda8d193ab
# ╠═6d0bfcd0-6879-4ea1-ba32-f4a74cbda890
# ╠═d67db15c-b26e-4f20-a882-32a45fc74411
# ╠═5887063d-e936-4b6b-b589-48718b4a3f33
# ╠═3fae0245-3823-41d6-99dc-e3ac7639fe6d
# ╠═faddb3c9-f059-495e-85c8-64fbb60080ff
# ╠═18d02e17-7864-4653-a40e-2b0ca47e8849
# ╠═59f33b66-8812-48d3-b50b-186bb033f442
# ╠═b27d40b9-1f2c-4c7c-b852-e8d549653117
# ╠═14f41921-3dbe-4817-935b-0731c0531624
# ╠═d0467970-7f6b-4fd4-98e5-5925398e32d9
# ╠═d0137c19-2928-4233-a398-4555cc11e155
# ╠═4c877223-6efb-417d-9272-63cac89af785
# ╠═3b29dc0e-e2c7-4694-9259-e3a82ed7670f
# ╠═ca3508b8-4fab-462f-ab55-9be8a041ebc4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
