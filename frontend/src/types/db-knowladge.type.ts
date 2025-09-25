export type Document = {
  icon: string;
  name: string;
  status: string;
  statusColor: string;
  size: string;
  owner: string;
  ownerAvatar: string;
  lastSync: string;
};

export type StatCard = {
  type: "stat";
  title: string;
  value: string | number;
  subtext?: string;
};

export type ProgressCard = {
  type: "progress";
  title: string;
  value: number; // percent
  subtext?: string;
};

export type GraphCard = {
  type: "graph";
  title: string;
  data: number[];
  subtext?: string;
};

export type ListCard = {
  type: "list";
  title: string;
  items: string[];
};

export type InfoCard = StatCard | ProgressCard | GraphCard | ListCard;

export type CollectionData = {
  title: string;
  icon: string;
  documents: Document[];
  cards: InfoCard[];
};
