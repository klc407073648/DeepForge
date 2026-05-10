export default function Home() {
  return (
    <main style={{ fontFamily: "sans-serif", padding: 32 }}>
      <h1>DeepForge Backend</h1>
      <p>Payload CMS is running.</p>
      <ul>
        <li>
          <a href="/admin">Admin</a>
        </li>
        <li>
          <a href="/api/posts">Posts API</a>
        </li>
      </ul>
    </main>
  );
}
