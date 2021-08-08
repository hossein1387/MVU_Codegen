
#*****************************************************************************
# Template file for auto code generation
# DO NOT MODIFY
#*****************************************************************************
#include "pito_def.h"

.globl _start
.globl _prog_end
.globl _fail

.section .text;
.section .text.init;
j reset_vector
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
    li      sp, 0x00003fc # set sp to the end of the memory
    la      t0, main
    csrw    mepc, t0
    mret

main:
    addi sp, sp, -4
    sw ra, 4(sp)
    jal __startup_code__
 --> FUNCCALL <--
    lw ra, 4(sp)
    addi sp, sp, 4
    j _prog_end

# in startup code, we need to set the following:
#   -> mtvec addresses
__startup_code__:
    # creating mtvec mask
    addi sp, sp, -4
    sw ra, 4(sp)
    jal enable_mvu_irq
    lui  a0, %hi(mvu_irq_handler)
    addi a0, a0, %lo(mvu_irq_handler )
    csrw mtvec, a0
    lw ra, 4(sp)
    addi sp, sp, 4
    ret

wait_for_mvu_irq:
    addi sp, sp, -24
    sw ra, 4(sp)
    sw s0, 8(sp)
    sw s1, 12(sp)
    sw s2, 16(sp)
    sw s3, 20(sp)
    sw s4, 24(sp)
wait_for_mvu_irq_loop:
    csrr t0, mcause
    srli t0, t0, 31
    addi t1, x0, 1
    # wait for mcause[31] interrupt to go high
    bne t0, t1, wait_for_mvu_irq_loop
    lw ra, 4(sp)
    lw s0, 8(sp)
    lw s1, 12(sp)
    lw s2, 16(sp)
    lw s3, 20(sp)
    lw s4, 24(sp)
    addi sp, sp, 24
    ret

mvu_irq_handler:
    # make sure global interrupt is disabled
    csrwi mstatus, 0x0
    # first things first, clear mvu intterupts pending bit while processing current irq.
    addi t1, x0, 1
    slli t1, t1, 16
    csrc mip, t1
    # do whatever to make MVU happy
    addi x0, x0, 0
    # we can now start processing incoming interrupts
    jal enable_mvu_irq
    mret

enable_mvu_irq:
    addi sp, sp, -4
    sw ra, 4(sp)
    # make sure global interrupt is enabled
    csrwi mstatus, 0x8
    # set MVU specific MIE bit aka mie[16]
    addi t0, x0, 1
    slli t0, t0, 16
    csrw mie, t0
    addi ra, sp, 0
    lw ra, 4(sp)
    addi sp, sp, 4
    ret

disable_mvu_irq:
    addi sp, sp, -4
    sw ra, 4(sp)
    # clear MVU specific MIE bit
    addi t0, x0, 1
    slli t0, t0, 16
    not t0, t0
    csrw mie, t0
    addi ra, sp, 0
    lw ra, 4(sp)
    addi sp, sp, 4
    ret

clear_mvu_pending_irq:
    addi sp, sp, -4
    sw ra, 4(sp)
    csrrci x0, mip, 0
    lw ra, 4(sp)
    addi sp, sp, 4
    ret

--> HERE <--

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

.section .data

