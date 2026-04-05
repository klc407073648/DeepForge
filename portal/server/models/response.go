package models

type ApiResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Code    int         `json:"code,omitempty"`
}

type UserInfoResponse struct {
	User        User     `json:"user"`
	Permissions []string `json:"permissions"`
}

type RefreshTokenRequest struct {
	RefreshToken string `json:"refreshToken" binding:"required"`
}

type RefreshTokenResponse struct {
	Token        string `json:"token"`
	RefreshToken string `json:"refreshToken"`
	ExpiresIn    int64  `json:"expiresIn"`
}

type UserListRequest struct {
	Page     int    `form:"page" binding:"required,min=1"`
	PageSize int    `form:"pageSize" binding:"required,min=1,max=100"`
	Username string `form:"username"`
}

type UserListResponse struct {
	Users    []User `json:"users"`
	Total    int64  `json:"total"`
	Page     int    `json:"page"`
	PageSize int    `json:"pageSize"`
}

type UpdateUserRoleRequest struct {
	UserID uint     `json:"userId" binding:"required"`
	Role   UserRole `json:"role" binding:"required"`
}
