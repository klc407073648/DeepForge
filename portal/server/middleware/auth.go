package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"go-auth-server/models"
	"go-auth-server/utils"
)

func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, models.ApiResponse{
				Success: false,
				Message: "缺少认证令牌",
				Code:    401,
			})
			c.Abort()
			return
		}

		// 检查Bearer前缀
		tokenParts := strings.SplitN(authHeader, " ", 2)
		if len(tokenParts) != 2 || tokenParts[0] != "Bearer" {
			c.JSON(http.StatusUnauthorized, models.ApiResponse{
				Success: false,
				Message: "认证令牌格式错误",
				Code:    401,
			})
			c.Abort()
			return
		}

		token := tokenParts[1]
		claims, err := utils.ParseToken(token)
		if err != nil {
			c.JSON(http.StatusUnauthorized, models.ApiResponse{
				Success: false,
				Message: "无效的认证令牌",
				Code:    401,
			})
			c.Abort()
			return
		}

		// 将用户信息存储到上下文中
		c.Set("userID", claims.UserID)
		c.Set("username", claims.Username)
		c.Set("role", claims.Role)

		c.Next()
	}
}


