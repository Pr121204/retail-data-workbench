import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // No proxy: the frontend calls http://localhost:8000 directly and the
  // backend allows http://localhost:5173 via CORS middleware.
});
