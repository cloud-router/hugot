//go:build XLA || ALL

package hugot

import (
	_ "github.com/gomlx/gomlx/backends/stablehlo"

	"github.com/knights-analytics/hugot/options"
)

func NewXLASession(opts ...options.WithOption) (*Session, error) {
	return newSession("XLA", opts...)
}
