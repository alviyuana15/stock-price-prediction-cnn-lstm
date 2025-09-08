<script lang="ts">
	import '../app.css';
	import Sidebar from '$lib/Sidebar.svelte';
	let { children } = $props();
	let isCollapsed = $state(false);
</script>

<div class="flex h-screen bg-gradient-to-br from-gray-800 to-gray-700">
	<Sidebar bind:isCollapsed />
	<div class="flex-1 overflow-y-auto px-6 py-6">
		{@render children()}
	</div>
</div>

<style>
	/* Tambahkan style berikut untuk memastikan konten utama memiliki tinggi minimum dan warna teks yang sesuai */
	.flex-1 {
		min-height: calc(100vh - 64px);
		color: white;
	}
</style>
