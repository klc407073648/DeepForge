import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const mediaItems = [
  {
    name: "hero-cover.png",
    type: "Image",
    size: "1.8 MB",
    usage: "官网头图"
  },
  {
    name: "product-guide.pdf",
    type: "Document",
    size: "860 KB",
    usage: "帮助中心"
  },
  {
    name: "dashboard-preview.webp",
    type: "Image",
    size: "420 KB",
    usage: "门户预览"
  }
];

export default function PortalMediaPage() {
  return (
    <div className="portal-page">
      <div className="portal-page-heading">
        <p>Media</p>
        <h2>媒体管理</h2>
        <span>维护内容封面、文档和门户素材。</span>
      </div>

      <div className="media-grid">
        {mediaItems.map((item) => (
          <Card key={item.name} className="media-card">
            <CardHeader>
              <div className="media-thumb">{item.type.slice(0, 1)}</div>
              <CardTitle>{item.name}</CardTitle>
              <CardDescription>
                {item.type} · {item.size}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="media-meta">
                <span>使用场景</span>
                <strong>{item.usage}</strong>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
