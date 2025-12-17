package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	fmt.Println("ATM Trader Service")
	fmt.Println("Version: 0.1.0")

	if len(os.Args) > 1 {
		fmt.Printf("Args: %v\n", os.Args[1:])
	}

	log.Println("Trader service starting...")
	// TODO: 实现交易服务逻辑
}
