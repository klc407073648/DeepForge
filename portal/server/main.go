package main

import (
	"log"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"

	"go-auth-server/handlers"
	"go-auth-server/middleware"
	"go-auth-server/models"
	"go-auth-server/services"
)

func main() {
	// 初始化数据库
	db, err := gorm.Open(sqlite.Open("auth.db"), &gorm.Config{})
	if err != nil {
		log.Fatal("数据库连接失败:", err)
	}

	// 自动迁移
	db.AutoMigrate(&models.User{})

	// 创建默认管理员用户
	createDefaultAdmin(db)

	// 初始化服务
	authService := services.NewAuthService(db)
	authHandler := handlers.NewAuthHandler(authService)

	// 初始化Gin
	r := gin.Default()

	// CORS配置
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:4200"}, // Angular开发服务器地址
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
	}))

	// API路由组
	api := r.Group("/api")
	{
		// 认证路由
		auth := api.Group("/auth")
		{
			auth.POST("/login", authHandler.Login)
			auth.POST("/register", authHandler.Register)
			auth.POST("/logout", authHandler.Logout)
			auth.POST("/refresh", authHandler.RefreshToken)
			auth.GET("/me", middleware.AuthMiddleware(), authHandler.GetCurrentUser)
		}

		// 用户管理路由
		users := api.Group("/users")
		users.Use(middleware.AuthMiddleware())
		{
			users.GET("/list", authHandler.GetUserList)
			users.PUT("/role", authHandler.UpdateUserRole)
		}
	}

	// 启动服务器
	log.Println("服务器启动在端口 :3000")
	r.Run(":3000")
}

func createDefaultAdmin(db *gorm.DB) {
	var count int64
	db.Model(&models.User{}).Count(&count)

	// if count == 0 {
	// 	// 创建管理员用户
	// 	hashedPasswordAdmin, _ := services.HashPassword("admin123")
	// 	admin := models.User{
	// 		Username: "admin",
	// 		Password: hashedPasswordAdmin,
	// 		Role:     models.RoleAdmin,
	// 	}
	// 	db.Create(&admin)
	// 	log.Println("默认管理员用户已创建: admin/admin123")

	// 	// 创建普通用户用于测试
	// 	hashedPasswordUser, _ := services.HashPassword("user123")
	// 	user := models.User{
	// 		Username: "user",
	// 		Password: hashedPasswordUser,
	// 		Role:     models.RoleUser,
	// 	}
	// 	db.Create(&user)
	// 	log.Println("默认普通用户已创建: user/user123")
	// }
}
