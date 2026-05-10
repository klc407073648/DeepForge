import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export type AuthUser = {
  id: string;
  name: string;
  email: string;
  role: "admin" | "editor";
};

type StoredUser = AuthUser & {
  password: string;
};

type RegisterInput = {
  name: string;
  email: string;
  password: string;
};

type AuthState = {
  currentUser: AuthUser | null;
  users: StoredUser[];
  hydrated: boolean;
  login: (email: string, password: string) => boolean;
  register: (input: RegisterInput) => { ok: boolean; message?: string };
  logout: () => void;
  setHydrated: (hydrated: boolean) => void;
};

const demoUsers: StoredUser[] = [
  {
    id: "user-demo-admin",
    name: "管理员",
    email: "admin@deepforge.dev",
    password: "123456",
    role: "admin"
  }
];

function toAuthUser(user: StoredUser): AuthUser {
  const { password: _password, ...safeUser } = user;
  return safeUser;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      currentUser: null,
      users: demoUsers,
      hydrated: false,
      login: (email, password) => {
        const user = get().users.find(
          (item) => item.email.toLowerCase() === email.toLowerCase() && item.password === password
        );

        if (!user) {
          return false;
        }

        set({ currentUser: toAuthUser(user) });
        return true;
      },
      register: (input) => {
        const exists = get().users.some((user) => user.email.toLowerCase() === input.email.toLowerCase());

        if (exists) {
          return {
            ok: false,
            message: "该邮箱已注册，请直接登录。"
          };
        }

        const user: StoredUser = {
          id: `user-${crypto.randomUUID()}`,
          name: input.name,
          email: input.email,
          password: input.password,
          role: "editor"
        };

        set((state) => ({
          users: [...state.users, user],
          currentUser: toAuthUser(user)
        }));

        return { ok: true };
      },
      logout: () => set({ currentUser: null }),
      setHydrated: (hydrated) => set({ hydrated })
    }),
    {
      name: "deepforge-auth",
      partialize: (state) => ({
        currentUser: state.currentUser,
        users: state.users
      }),
      storage: createJSONStorage(() => localStorage),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated(true);
      }
    }
  )
);
