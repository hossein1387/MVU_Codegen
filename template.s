#*****************************************************************************
# Template file for auto code generation
# DO NOT MODIFY
#*****************************************************************************
#include "pito_def.h"

jal sp, enable_mvu_irq
jal sp, __startup_code__
// --> FUNCCALL <--
jal sp, prog_end


// in startup code, we need to set the following:
//   -> mtvec addresses
__startup_code__:
    // addi x1, x0, pito_mtvec_mask
    // creating mtvec mask
    lui  a0, %hi(mvu_irq_handler)
    addi a0, a0, %lo(mvu_irq_handler )
    csrw mtvec, a0
    addi ra, sp, 0
    ret

wait_for_mvu_irq:
    csrr t0, mcause
    srli t0, t0, 31
    addi t1, x0, 1
    // wait for mcause[31] interrupt to go high
    bne t0, t1, wait_for_mvu_irq
    addi ra, t3, 0
    ret

mvu_irq_handler:
    // make sure global interrupt is disabled
    csrwi mstatus, 0x0
    // first things first, clear mvu intterupts pending bit while processing current irq.
    addi t1, x0, 1
    slli t1, t1, 16
    csrc mip, t1
    // do whatever to make MVU happy
    addi x0, x0, 0
    // we can now start processing incoming interrupts
    addi gp, sp, 0
    jal sp, enable_mvu_irq
    addi ra, gp, 0
    mret

enable_mvu_irq:
    // make sure global interrupt is enabled
    csrwi mstatus, 0x8
    // set MVU specific MIE bit aka mie[16]
    addi t0, x0, 1
    slli t0, t0, 16
    csrw mie, t0
    addi ra, sp, 0
    ret

disable_mvu_irq:
    // clear MVU specific MIE bit
    addi t0, x0, 1
    slli t0, t0, 16
    not t0, t0
    csrw mie, t0
    addi ra, sp, 0
    ret

clear_mvu_pending_irq:
    csrrci x0, mip, 0
    ret

// --> HERE <--

// Done with our awesome program!
prog_end:
    lui a0,0x10000000>>12
    addi  a1,zero,'O'
    addi  a2,zero,'K'
    addi  a3,zero,'\n'
    sw  a1,0(a0)
    sw  a2,0(a0)
    sw  a3,0(a0)
    ebreak
