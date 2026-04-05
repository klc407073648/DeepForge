package config

import (
	"os"
)

type Config struct {
	Port        string
	JWTSecret   string
	DatabaseURL string
	CORSOrigins []string
}

func LoadConfig() *Config {
	return &Config{
		Port:        getEnv("PORT", "3000"),
		JWTSecret:   getEnv("JWT_SECRET", "your-secret-key"),
		DatabaseURL: getEnv("DATABASE_URL", "auth.db"),
		CORSOrigins: []string{
			getEnv("CORS_ORIGIN", "http://localhost:4200"),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}


