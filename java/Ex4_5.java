public class Ex4_5 {
    public static void main(String[] args){
        int x = 0;
        while(x<=10) {
            int y = 0;
            // 매 시작마다 y를 0으로 초기화해서 "*"을 1개씩 늘리면서 찍을 수 있음
            while(y<=x){
            System.out.print("*");
            y++;
            } System.out.println();
            x++;
        }
    }
}
