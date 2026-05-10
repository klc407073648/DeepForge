import type { CollectionConfig } from "payload";

export const Posts: CollectionConfig = {
  slug: "posts",
  admin: {
    useAsTitle: "title"
  },
  access: {
    read: () => true
  },
  fields: [
    {
      name: "title",
      type: "text",
      required: true
    },
    {
      name: "slug",
      type: "text",
      required: true,
      unique: true,
      index: true
    },
    {
      name: "status",
      type: "select",
      defaultValue: "draft",
      options: [
        {
          label: "Draft",
          value: "draft"
        },
        {
          label: "Published",
          value: "published"
        }
      ]
    },
    {
      name: "excerpt",
      type: "textarea"
    },
    {
      name: "cover",
      type: "upload",
      relationTo: "media"
    },
    {
      name: "content",
      type: "richText"
    }
  ]
};
