package models

import "time"

type UserRole string

const (
	RoleAdmin UserRole = "admin"
	RoleUser  UserRole = "user"
)

type User struct {
	ID          uint       `json:"id" gorm:"primaryKey"`
	Username    string     `json:"username" gorm:"uniqueIndex;not null"`
	Password    string     `json:"-" gorm:"not null"` // 不在JSON中返回
	Role        UserRole   `json:"role" gorm:"default:'user'"`
	Avatar      string     `json:"avatar,omitempty"`
	LastLoginAt *time.Time `json:"lastLoginAt,omitempty"`
	CreatedAt   time.Time  `json:"createdAt"`
	UpdatedAt   time.Time  `json:"updatedAt"`
}

type LoginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type LoginResponse struct {
	User         User   `json:"user"`
	Token        string `json:"token"`
	RefreshToken string `json:"refreshToken,omitempty"`
	ExpiresIn    int64  `json:"expiresIn"`
}

type RegisterRequest struct {
	Username string `json:"username" binding:"required,min=3,max=20"`
	Password string `json:"password" binding:"required,min=8"`
}

type RegisterResponse struct {
	User         User   `json:"user"`
	Token        string `json:"token"`
	RefreshToken string `json:"refreshToken,omitempty"`
	ExpiresIn    int64  `json:"expiresIn"`
}
