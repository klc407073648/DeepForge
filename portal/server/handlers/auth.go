package handlers

import (
	"go-auth-server/models"
	"go-auth-server/services"
	"net/http"

	"github.com/gin-gonic/gin"
)

type AuthHandler struct {
	authService *services.AuthService
}

func NewAuthHandler(authService *services.AuthService) *AuthHandler {
	return &AuthHandler{authService: authService}
}

// 用户登录
func (h *AuthHandler) Login(c *gin.Context) {
	var req models.LoginRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: "请求参数错误",
			Code:    400,
		})
		return
	}

	loginResp, err := h.authService.Login(&req)
	if err != nil {
		c.JSON(http.StatusUnauthorized, models.ApiResponse{
			Success: false,
			Message: err.Error(),
			Code:    401,
		})
		return
	}

	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "登录成功",
		Data:    loginResp,
		Code:    200,
	})
}

// 获取当前用户信息
func (h *AuthHandler) GetCurrentUser(c *gin.Context) {
	userID, exists := c.Get("userID")
	if !exists {
		c.JSON(http.StatusUnauthorized, models.ApiResponse{
			Success: false,
			Message: "未授权",
			Code:    401,
		})
		return
	}

	userInfo, err := h.authService.GetUserInfo(userID.(uint))
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ApiResponse{
			Success: false,
			Message: "获取用户信息失败",
			Code:    500,
		})
		return
	}

	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "获取成功",
		Data:    userInfo,
		Code:    200,
	})
}

// 刷新令牌
func (h *AuthHandler) RefreshToken(c *gin.Context) {
	var req models.RefreshTokenRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: "请求参数错误",
			Code:    400,
		})
		return
	}

	refreshResp, err := h.authService.RefreshToken(req.RefreshToken)
	if err != nil {
		c.JSON(http.StatusUnauthorized, models.ApiResponse{
			Success: false,
			Message: err.Error(),
			Code:    401,
		})
		return
	}

	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "令牌刷新成功",
		Data:    refreshResp,
		Code:    200,
	})
}

// 用户登出
func (h *AuthHandler) Logout(c *gin.Context) {
	// 这里可以实现令牌黑名单机制
	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "登出成功",
		Code:    200,
	})
}

// 用户注册
func (h *AuthHandler) Register(c *gin.Context) {
	var req models.RegisterRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: "请求参数错误",
			Code:    400,
		})
		return
	}

	registerResp, err := h.authService.Register(&req)
	if err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: err.Error(),
			Code:    400,
		})
		return
	}

	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "注册成功",
		Data:    registerResp,
		Code:    200,
	})
}

// 获取用户列表
func (h *AuthHandler) GetUserList(c *gin.Context) {
	var req models.UserListRequest

	if err := c.ShouldBindQuery(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: "请求参数错误: " + err.Error(),
			Code:    400,
		})
		return
	}

	// 获取当前用户信息
	userID, exists := c.Get("userID")
	if !exists {
		c.JSON(http.StatusUnauthorized, models.ApiResponse{
			Success: false,
			Message: "未授权: 缺少userID",
			Code:    401,
		})
		return
	}

	userRole, exists := c.Get("role")
	if !exists {
		c.JSON(http.StatusUnauthorized, models.ApiResponse{
			Success: false,
			Message: "未授权: 缺少role",
			Code:    401,
		})
		return
	}

	isAdmin := userRole.(string) == string(models.RoleAdmin)

	userList, err := h.authService.GetUserList(&req, userID.(uint), isAdmin)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.ApiResponse{
			Success: false,
			Message: "获取用户列表失败: " + err.Error(),
			Code:    500,
		})
		return
	}

	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "获取成功",
		Data:    userList,
		Code:    200,
	})
}

// 更新用户角色
func (h *AuthHandler) UpdateUserRole(c *gin.Context) {
	var req models.UpdateUserRoleRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: "请求参数错误: " + err.Error(),
			Code:    400,
		})
		return
	}

	// 获取当前用户ID（操作者）
	operatorID, exists := c.Get("userID")
	if !exists {
		c.JSON(http.StatusUnauthorized, models.ApiResponse{
			Success: false,
			Message: "未授权",
			Code:    401,
		})
		return
	}

	// 验证操作者是否为管理员
	operatorRole, exists := c.Get("role")
	if !exists || operatorRole.(string) != string(models.RoleAdmin) {
		c.JSON(http.StatusForbidden, models.ApiResponse{
			Success: false,
			Message: "权限不足：只有管理员可以修改用户角色",
			Code:    403,
		})
		return
	}

	err := h.authService.UpdateUserRole(req.UserID, req.Role, operatorID.(uint))
	if err != nil {
		c.JSON(http.StatusBadRequest, models.ApiResponse{
			Success: false,
			Message: err.Error(),
			Code:    400,
		})
		return
	}

	c.JSON(http.StatusOK, models.ApiResponse{
		Success: true,
		Message: "角色更新成功",
		Code:    200,
	})
}
