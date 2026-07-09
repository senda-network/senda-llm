import { cn } from '../lib/utils';

type SendaWordmarkProps = {
  className?: string;
};

export function SendaWordmark({ className }: SendaWordmarkProps) {
  return (
    <span className={cn('whitespace-nowrap', className)}>
      <span className="text-primary">Sen</span>
      da
    </span>
  );
}

SendaWordmark.displayName = 'SendaWordmark';
