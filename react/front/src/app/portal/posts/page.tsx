import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const posts = [
  {
    title: "DeepForge 产品发布计划",
    status: "Published",
    author: "管理员",
    updatedAt: "2026-05-10",
    views: "12.8k"
  },
  {
    title: "Payload CMS 内容模型设计",
    status: "Draft",
    author: "内容编辑",
    updatedAt: "2026-05-08",
    views: "846"
  },
  {
    title: "门户网站信息架构说明",
    status: "Review",
    author: "运营团队",
    updatedAt: "2026-05-06",
    views: "2.1k"
  }
];

export default function PortalPostsPage() {
  return (
    <div className="portal-page">
      <div className="portal-page-heading">
        <p>Posts</p>
        <h2>文章管理</h2>
        <span>管理发布内容、草稿和审核状态。</span>
      </div>

      <Card>
        <CardHeader className="portal-card-header">
          <div>
            <CardTitle>内容列表</CardTitle>
            <CardDescription>后续可对接 Payload `posts` 集合。</CardDescription>
          </div>
          <Button>新建文章</Button>
        </CardHeader>
        <CardContent>
          <div className="portal-table">
            <div className="portal-table-row header">
              <span>标题</span>
              <span>状态</span>
              <span>作者</span>
              <span>浏览量</span>
              <span>更新时间</span>
            </div>
            {posts.map((post) => (
              <div key={post.title} className="portal-table-row">
                <strong>{post.title}</strong>
                <span className={`status-badge status-${post.status.toLowerCase()}`}>{post.status}</span>
                <span>{post.author}</span>
                <span>{post.views}</span>
                <span>{post.updatedAt}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
