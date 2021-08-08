
j reset_vector
j _prog_end

reset_vector:
    li      x1, 0
    li      x4, 0
    li      x5, 0
    li      x6, 0
    li      x7, 0
    li      x8, 0
    li      x9, 0
    li      x10, 0
    li      x11, 0
    li      x12, 0
    li      x13, 0
    li      x14, 0
    li      x15, 0
    li      x16, 0
    li      x17, 0
    li      x18, 0
    li      x19, 0
    li      x20, 0
    li      x10, 0
    li      x21, 0
    li      x22, 0
    li      x23, 0
    li      x24, 0
    li      x25, 0
    li      x26, 0
    li      x27, 0
    li      x28, 0
    li      x29, 0
    li      x30, 0
    li      x31, 0
    la      t0, main
    # csrw    mepc, t0
    ret

main:
    jal sp, enable_mvu_irq
    jal sp, __startup_code__
    jal _init_matmul_0
    jal sp, MatMul_0


# in startup code, we need to set the following:
#   -> mtvec addresses
__startup_code__:
    # addi x1, x0, pito_mtvec_mask
    # creating mtvec mask
    lui  a0, %hi(mvu_irq_handler)
    addi a0, a0, %lo(mvu_irq_handler )
    # csrw mtvec, a0
    addi ra, sp, 0
    ret

wait_for_mvu_irq:
    csrr t0, mcause
    srli t0, t0, 31
    addi t1, x0, 1
    # wait for mcause[31] interrupt to go high
    bne t0, t1, wait_for_mvu_irq
    addi ra, t3, 0
    ret

mvu_irq_handler:
    # make sure global interrupt is disabled
    # csrwi mstatus, 0x0
    # first things first, clear mvu intterupts pending bit while processing current irq.
    addi t1, x0, 1
    slli t1, t1, 16
    csrc mip, t1
    # do whatever to make MVU happy
    addi x0, x0, 0
    # we can now start processing incoming interrupts
    addi gp, sp, 0
    jal sp, enable_mvu_irq
    addi ra, gp, 0
    mret

enable_mvu_irq:
    # make sure global interrupt is enabled
    # csrwi mstatus, 0x8
    # set MVU specific MIE bit aka mie[16]
    addi t0, x0, 1
    slli t0, t0, 16
    # csrw mie, t0
    addi ra, sp, 0
    ret

disable_mvu_irq:
    # clear MVU specific MIE bit
    addi t0, x0, 1
    slli t0, t0, 16
    not t0, t0
    # csrw mie, t0
    addi ra, sp, 0
    ret

clear_mvu_pending_irq:
    csrrci x0, mip, 0
    ret

MatMul_0:
    jal sp, MatMul_0_init
    jal sp, MatMul_0_loop
    ret

MatMul_0_init:
	addi  t1, x0, 0
	addi  t2, x0, 2
	add   t1, t1, t2
	addi  t2, x0, 2
	slli  t3, t2, 6
	add   t1, t1, t3
	addi  t2, x0, 2
	slli  t3, t2, 12
	add   t1, t1, t3
	addi t1, x0, 1
	slli t1, t1, 30
	addi t1, t1, 4
	ret

MatMul_0_loop:
    addi t0, x0, 4
loop:
	addi t1, 1024
    jal t3, wait_for_mvu_irq
    addi t0,t0, -1 

    bne t0, x0, loop
    ret

# Done with our awesome program!
_prog_end:
    lui a0,0x10000000>>12
    addi  a1,zero,'O'
    addi  a2,zero,'K'
    addi  a3,zero,'\n'
    sw  a1,0(a0)
    sw  a2,0(a0)
    sw  a3,0(a0)
    ebreak

_fail:
    lui a0,0x10000000>>12
    addi  a1,zero,'N'
    addi  a2,zero,'O'
    addi  a3,zero,'K'
    addi  a4,zero,'\n'
    sw  a1,0(a0)
    sw  a2,0(a0)
    sw  a3,0(a0)
    sw  a4,0(a0)
    ebreak

