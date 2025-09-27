import { cn } from "@/lib/utils"

function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="skeleton"
      className={cn("animate-pulse rounded-md", className)}
      style={{
        background: 'var(--primary-8, rgba(19, 245, 132, 0.08))'
      }}
      {...props}
    />
  )
}

export { Skeleton }
