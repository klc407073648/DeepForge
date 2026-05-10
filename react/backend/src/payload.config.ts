import { postgresAdapter } from "@payloadcms/db-postgres";
import { payloadCloudPlugin } from "@payloadcms/payload-cloud";
import { lexicalEditor } from "@payloadcms/richtext-lexical";
import path from "path";
import { buildConfig } from "payload";
import { fileURLToPath } from "url";
import { Media } from "./collections/Media";
import { Posts } from "./collections/Posts";
import { Users } from "./collections/Users";

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

export default buildConfig({
  admin: {
    user: Users.slug
  },
  collections: [Users, Media, Posts],
  cors: [process.env.FRONTEND_URL ?? "http://localhost:3000"],
  csrf: [process.env.FRONTEND_URL ?? "http://localhost:3000"],
  db: postgresAdapter({
    pool: {
      connectionString: process.env.DATABASE_URI
    }
  }),
  editor: lexicalEditor({}),
  endpoints: [
    {
      path: "/dashboard/stats",
      method: "get",
      handler: async () =>
        Response.json({
          users: 1280,
          revenue: 101000,
          conversionRate: 12.8
        })
    },
    {
      path: "/dashboard/revenue",
      method: "get",
      handler: async () =>
        Response.json([
          { month: "1月", revenue: 12000 },
          { month: "2月", revenue: 18000 },
          { month: "3月", revenue: 15000 },
          { month: "4月", revenue: 24000 },
          { month: "5月", revenue: 32000 }
        ])
    }
  ],
  plugins: [payloadCloudPlugin()],
  secret: process.env.PAYLOAD_SECRET ?? "development-secret",
  typescript: {
    outputFile: path.resolve(dirname, "payload-types.ts")
  }
});
