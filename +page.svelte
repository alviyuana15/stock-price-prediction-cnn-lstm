<div
	class="animate-fade-in mx-auto mt-10 max-w-4xl rounded-xl border border-gray-700 bg-gradient-to-br from-gray-900 to-gray-800 p-8 shadow-xl"
>
	<div class="flex items-start gap-4">
		<div class="rounded-full bg-gray-800 p-3">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				class="h-10 w-10 text-indigo-400"
				fill="none"
				viewBox="0 0 24 24"
				stroke="currentColor"
				stroke-width="2"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
				/>
			</svg>
		</div>
		<div>
			<h1 class="mb-2 text-3xl font-bold text-gray-100 md:text-4xl">
				Selamat Datang di <span class="text-indigo-400">Dashboard Analisis Saham</span>
			</h1>
			<p class="mb-6 text-lg text-gray-300">
				Mulai analisis prediktif saham Anda dengan memilih menu di sidebar.
			</p>

			<div class="animate-pulse-slow rounded-lg border border-gray-700 bg-gray-800 p-4">
				<div class="flex items-center gap-3">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-5 w-5 text-indigo-400"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
						/>
					</svg>
					<p class="text-sm text-indigo-200">
						Tips: Gunakan fitur prediksi untuk melihat proyeksi saham ke depan
					</p>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.animate-fade-in {
		animation: fadeIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
	}

	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateY(20px) scale(0.98);
		}
		to {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
	}

	.animate-pulse-slow {
		animation: pulseSlow 4s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}

	@keyframes pulseSlow {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.8;
		}
	}
</style>
